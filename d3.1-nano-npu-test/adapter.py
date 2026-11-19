"""Offload the short CPU/BF16 text path to edge's native RKNN kernels."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import torch
from native_mlp import NativeMlp


class ShortTextNPU:
    """Single inference owner. Its gate and up projections run on separate NPU cores."""

    def __init__(self, model):
        self.restores = []
        self.counts = Counter()
        self.rows = Counter()
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.kernels = None
        self.mlps = self.projections = 0
        try:
            core = {}
            values = torch.arange(65536, dtype=torch.int32).short().view(torch.bfloat16)
            table = SimpleNamespace(silu=torch.nn.functional.silu(values).contiguous())
            self.kernels = NativeMlp(SimpleNamespace(core=core, pool=self.pool), table, "libnative_short.so")
            for layer in model.backbone.language_model.layers:
                mlp = layer.mlp
                members = (mlp.gate_proj, mlp.up_proj, mlp.down_proj)
                if any(m.bias is not None for m in members) or type(mlp.act_fn).__name__ not in ("SiLUActivation", "SiLU"):
                    raise ValueError("Unsupported MLP activation or bias")
                core[id(mlp.gate_proj)] = 1
                core[id(mlp.up_proj)] = 2
                self.install(mlp, members, "mlp")
                self.mlps += 1
                for name, module in layer.named_modules():
                    if name.startswith("mlp.") or not isinstance(module, torch.nn.Linear):
                        continue
                    if module.bias is None and module.weight.numel() >= 262144:
                        self.install(module, (module,), "projection")
                        self.projections += 1
        except BaseException:
            self.close()
            raise

    def install(self, module, members, kind):
        original = module.forward
        weights = [(m.weight, m.weight._version) for m in members]
        self.restores.append((module, original))
        module.forward = partial(self.forward, kind, module, members, weights, original)

    def forward(self, kind, module, members, weights, original, x):
        count = x.numel() // x.shape[-1]
        if (torch.is_grad_enabled() or module.training or x.device.type != "cpu" or x.dtype != torch.bfloat16
                or x.ndim != 3 or x.shape[0] != 1 or not 0 < count <= 128
                or any(m.weight is not w or w._version != v for m, (w, v) in zip(members, weights))):
            self.counts["cpu_guard"] += 1
            return original(x)
        try:
            if kind == "mlp":
                output = self.kernels.run(members, x)
            else:
                output = self.kernels.execute(module, x)
        except ValueError:
            self.counts["cpu_range"] += 1
            return original(x)
        self.rows[count] += 1
        self.counts[kind] += 1
        self.counts["successful_npu_matmuls"] += len(members)
        return output

    def close(self):
        for module, original in reversed(self.restores):
            module.forward = original
        self.restores.clear()
        self.pool.shutdown(wait=True)
        if self.kernels is not None:
            self.kernels.close()
