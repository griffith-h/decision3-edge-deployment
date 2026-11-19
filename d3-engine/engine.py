"""Single-owner text inference for D3's Qwen3.5 CPU checkpoints on RK3576."""

import copy
import threading
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import torch
from adapter import ShortTextNPU


class TiledNPU(ShortTextNPU):
    """Only token-independent GEMMs may cross batch and sequence boundaries."""

    def forward(self, kind, module, members, weights, original, x):
        if (torch.is_grad_enabled() or module.training or x.ndim not in (2, 3) or not x.numel()
                or x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.shape[-1] != members[0].in_features
                or any(m.weight is not w or w._version != v or w.dtype != x.dtype or w.device != x.device
                       for m, (w, v) in zip(members, weights))):
            self.counts["linear_fallback"] += 1
            return original(x)
        flat = x.reshape(-1, x.shape[-1])
        width = members[-1].out_features
        out = torch.empty((len(flat), width), dtype=x.dtype) if len(flat) > 128 else None
        start_time = time.perf_counter()
        try:
            for start in range(0, len(flat), 128):
                part = flat[start:start+128]
                piece = self.kernels.run(members, part) if kind == "mlp" else self.kernels.execute(module, part)
                self.counts["npu_matmuls"] += len(members)
                if out is None:
                    return piece.reshape(*x.shape[:-1], width)
                out[start:start+len(part)].copy_(piece)
        except ValueError:
            # A range failure on any tile invalidates the entire operation.
            self.counts["linear_range_fallback"] += 1
        else:
            return out.reshape(*x.shape[:-1], width)
        finally:
            self.counts["npu_bridge_seconds"] += time.perf_counter() - start_time
        return original(x)


class QuestionBranches:
    """Hold one immutable state cache per request and bound the live question activations."""

    def __init__(self, source, counts, rows, checkpoint_state_kernels=False):
        self.source = copy.copy(source)
        if not checkpoint_state_kernels:
            from state_kernels import StateKernels
            self.source.qwen = StateKernels(source.qwen, counts, Path(__file__).with_name("libd3_state.so"))
        self.rows = rows
        self.counts = counts

    def state(self, embeds, rope):
        start = time.perf_counter()
        cache = []
        hidden = embeds
        for layer, softmax in self.source.stack:
            hidden, entry = self.source._layer(layer, softmax, hidden, rope, None, None)
            if not softmax:
                tail, final_state = entry
                # A view of three tokens otherwise retains the full state's QKV projection.
                excess = tail.untyped_storage().nbytes() - tail.numel() * tail.element_size()
                if excess > 0:
                    entry = tail.clone(memory_format=torch.contiguous_format), final_state
                    self.counts["conv_tail_excess_backing_bytes"] += excess
            cache.append(entry)
        self.counts["state_passes"] += 1
        self.counts["state_seconds"] += time.perf_counter() - start
        return cache

    def questions(self, cache, embeds, lengths, rope):
        start = time.perf_counter()
        hidden = []
        for i in range(0, len(lengths), self.rows):
            group = lengths[i:i+self.rows]
            width = max(group)
            part = slice(i, i + len(group))
            hidden.append(self.source.questions(cache, embeds[part, :width], group,
                                                tuple(t[part, :width] for t in rope)))
            self.counts["question_batches"] += 1
        self.counts["question_seconds"] += time.perf_counter() - start
        return torch.cat(hidden, dim=0)


class D3Engine:
    """Own one loaded model and its native buffers. Call system_one through this object."""

    _claim_lock = threading.Lock()

    def __init__(self, model, *, question_rows=4, attention="auto", checkpoint_state_kernels=False):
        if type(question_rows) is not int or question_rows < 1:
            raise ValueError("question_rows must be a positive integer")
        if type(checkpoint_state_kernels) is not bool:
            raise ValueError("checkpoint_state_kernels must be a boolean")
        if attention == "auto":
            attention = "fp32" if model.attention_mode == "prefix_bidir" else "checkpoint"
        if attention not in ("checkpoint", "fp32"):
            raise ValueError("attention must be auto, checkpoint or fp32")
        if attention == "fp32" and model.attention_mode != "prefix_bidir":
            raise ValueError("FP32 attention is implemented on the prefix_bidir path")
        if (model.backbone_type != "qwen3_5" or model.attention_mode not in ("prefix_bidir", "noncausal_full_attention")
                or torch.device(model.device).type != "cpu" or model.readout_dtype != "float32"
                or model.backbone.training):
            raise ValueError("expected a CPU inference-mode Qwen3.5 D3 checkpoint with FP32 readout")
        state_once = getattr(model, "state_once", None)
        if model.attention_mode == "prefix_bidir" and type(state_once).__name__ != "StateOnce":
            raise ValueError("prefix_bidir needs the released plain StateOnce implementation")
        self.model = model
        self.lock = threading.Lock()
        self.resources = ExitStack()
        self.closed = False
        self.stages = Counter()
        try:
            with self._claim_lock:
                if getattr(model, "_rk3576_owner", None) is not None:
                    raise ValueError("this checkpoint already has an active engine")
                model._rk3576_owner = self
            self.resources.callback(delattr, model, "_rk3576_owner")
            self.linear = TiledNPU(model)
            self.resources.callback(self.linear.close)
            if model.attention_mode == "prefix_bidir":
                branches = QuestionBranches(state_once, self.stages, question_rows, checkpoint_state_kernels)
                if attention == "fp32":
                    from cpu_attention import bind_attention
                    self.resources.callback(bind_attention(branches.source, self.stages))
                model.state_once = branches
                self.resources.callback(setattr, model, "state_once", state_once)
        except BaseException:
            self.resources.close()
            raise

    @property
    def counts(self):
        return self.linear.counts + self.stages

    def system_one(self, *, state, questions):
        with self.lock:
            if self.closed:
                raise RuntimeError("D3 engine is closed")
            return self.model.system_one(state=state, questions=questions)

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
