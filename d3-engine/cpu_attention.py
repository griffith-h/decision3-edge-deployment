"""CPU SDPA for one model's prefix attention, plus a same-tensor diagnostic."""

import time
from collections import Counter
from contextlib import contextmanager
from functools import wraps

import torch
from torch.overrides import TorchFunctionMode


class AttentionPrecision(TorchFunctionMode):
    def __init__(self, counts, replay):
        super().__init__()
        self.counts = counts
        self.replay = replay

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is not torch.nn.functional.scaled_dot_product_attention:
            return func(*args, **kwargs)
        q, k, v, *rest = args
        mask = rest[0] if rest else kwargs.get("attn_mask")
        if (torch.is_grad_enabled() or any(x.dtype != torch.bfloat16 or x.device.type != "cpu" for x in (q, k, v))
                or (mask is not None and mask.dtype not in (torch.bool, torch.float32))):
            self.counts["sdpa_fallback"] += 1
            return func(*args, **kwargs)
        if self.replay:
            start = time.perf_counter()
            original = func(*args, **kwargs)
            self.counts["sdpa_bf16_s"] += time.perf_counter() - start
        start = time.perf_counter()
        other = func(q.float(), k.float(), v.float(), *rest, **kwargs).bfloat16()
        self.counts["sdpa_calls"] += 1
        self.counts["sdpa_fp32_s"] += time.perf_counter() - start
        if not self.replay:
            return other
        diff = (original.float() - other.float()).abs().max().item()
        self.counts["sdpa_output_max_abs"] = max(self.counts["sdpa_output_max_abs"], diff)
        return original


def bind_attention(source, counts, *, replay=False):
    original = source._attention
    overridden = "_attention" in vars(source)

    @wraps(original)
    def attention(*args, **kwargs):
        with AttentionPrecision(counts, replay):
            return original(*args, **kwargs)

    source._attention = attention
    def restore():
        if overridden:
            source._attention = original
        else:
            del source._attention
    return restore


@contextmanager
def profile(engine):
    source = engine.model.state_once.source
    counts = Counter()
    restore = bind_attention(source, counts, replay=True)
    starts, handles = {}, []

    def timed(module, inputs, output):
        counts["small_linear_s"] += time.perf_counter() - starts.pop(id(module))

    def start(module, inputs):
        starts[id(module)] = time.perf_counter()

    try:
        for layer in engine.model.backbone.language_model.layers:
            delta = getattr(layer, "linear_attn", None)
            if delta is not None:
                for m in (delta.in_proj_a, delta.in_proj_b):
                    handles.append(m.register_forward_pre_hook(start))
                    handles.append(m.register_forward_hook(timed))
        yield counts
    finally:
        for handle in handles:
            handle.remove()
        restore()
