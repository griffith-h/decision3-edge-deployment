#!/usr/bin/env python3
"""Serve pinned d3 with faster ARM CPU Linear/Conv kernels.

Keep BF16 checkpoint tensors and activation boundaries. Compute Linear and
Conv operations in FP32 using cached FP32 weights, then round outputs to the
original activation dtype. Probability parity must be measured before use.
"""
import argparse
import os
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.nn.functional as functional


def linear_forward(self, inputs):
    return functional.linear(inputs.float(), self._cpu_fp32_weight,
                             self._cpu_fp32_bias).to(inputs.dtype)


def conv_forward(self, inputs):
    return self._conv_forward(inputs.float(), self._cpu_fp32_weight,
                             self._cpu_fp32_bias).to(inputs.dtype)


def optimize(model):
    count = 0
    cache_bytes = 0
    for module in model.backbone.modules():
        if not isinstance(module, (torch.nn.Linear, torch.nn.Conv1d,
                                   torch.nn.Conv2d, torch.nn.Conv3d)):
            continue
        if module.weight.device.type != "cpu" or module.weight.dtype != torch.bfloat16:
            continue
        weight = module.weight.detach().float()
        bias = None if module.bias is None else module.bias.detach().float()
        module.register_buffer("_cpu_fp32_weight", weight, persistent=False)
        module.register_buffer("_cpu_fp32_bias", bias, persistent=False)
        function = linear_forward if isinstance(module, torch.nn.Linear) else conv_forward
        module.forward = MethodType(function, module)
        count += 1
        cache_bytes += weight.numel() * weight.element_size()
        if bias is not None:
            cache_bytes += bias.numel() * bias.element_size()
    return count, cache_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    args = parser.parse_args()
    sys.path.insert(0, str(args.model))
    from d3_runtime import D3
    from d3_server import serve
    model = D3.from_pretrained(args.model, device="cpu", verify="full")
    model.model_name = "vllm-sr/d3-edge"
    count, size = optimize(model)
    print(f"Optimized {count} CPU layers; cached FP32 weights: {size / 2**20:.1f} MiB", flush=True)
    server = serve(model, args.host, args.port,
                   api_key=os.environ.get("DECISION_API_KEY") or None)
    print(f"d3 CPU optimized server ready on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
