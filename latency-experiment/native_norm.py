"""Scoped BF16 CPU norm fusion; unsupported dtypes/devices use stock forward."""
import ctypes
from pathlib import Path
import types
import torch

def install_norm(model, stats=None):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    lib = ctypes.CDLL(str(Path(__file__).with_name('libd3_native_norm.so')))
    fn = lib.d3_norm
    fn.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
    fn.restype = None
    originals = []
    for module in model.modules():
        if type(module) not in (qwen.Qwen3_5RMSNorm, qwen.Qwen3_5RMSNormGated):
            continue
        original = module.forward
        kind = int(type(module) is qwen.Qwen3_5RMSNormGated)
        eps = module.variance_epsilon if kind else module.eps
        def forward(self, x, gate=None, _original=original, _kind=kind, _eps=eps):
            if (torch.is_grad_enabled() or x.device.type != 'cpu' or x.dtype != torch.bfloat16
                    or self.weight.device.type != 'cpu' or self.weight.dtype != torch.bfloat16
                    or x.shape[-1] != self.weight.numel()
                    or (_kind and (gate is None or gate.device != x.device or gate.dtype != x.dtype
                                   or gate.shape != x.shape))):
                return _original(x, gate) if _kind else _original(x)
            inputs, weight = x.contiguous(), self.weight.contiguous()
            gating = gate.contiguous() if _kind else None
            output = torch.empty_like(inputs)
            fn(inputs.data_ptr(), weight.data_ptr(), gating.data_ptr() if _kind else None,
               output.data_ptr(), inputs.numel()//inputs.shape[-1], inputs.shape[-1], _eps, _kind)
            if stats is not None:
                stats['native_calls'] = stats.get('native_calls', 0) + 1
                stats['norm_calls'] = stats.get('norm_calls', 0) + 1
            return output
        originals.append((module, original))
        module.forward = types.MethodType(forward, module)
    def restore():
        for module, original in originals:
            module.forward = original
    return restore


def install_native_conv(stats=None):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    lib = ctypes.CDLL(str(Path(__file__).with_name('libd3_native_norm.so')))
    fn = lib.d3_conv4
    fn.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*3
    fn.restype = None
    original = qwen.causal_conv1d_fn
    def forward(x, weight, bias=None, activation=None, **kwargs):
        if (torch.is_grad_enabled() or x.device.type != 'cpu' or x.dtype != torch.bfloat16
                or x.ndim != 3 or weight.ndim != 2 or weight.shape != (x.shape[1], 4)
                or weight.device != x.device or weight.dtype != x.dtype
                or (bias is not None and (bias.device != x.device or bias.dtype != x.dtype
                                         or bias.shape != (x.shape[1],)))
                or activation not in (None, 'silu', 'swish')):
            return original(x, weight, bias, activation=activation, **kwargs)
        inputs, w = x.contiguous(), weight.contiguous()
        b = bias.contiguous() if bias is not None else None
        output = torch.empty_like(inputs)
        fn(inputs.data_ptr(), w.data_ptr(), b.data_ptr() if b is not None else None,
           output.data_ptr(), *inputs.shape)
        if stats is not None:
            stats['native_calls'] = stats.get('native_calls', 0) + 1
            stats['conv_calls'] = stats.get('conv_calls', 0) + 1
        return torch.nn.functional.silu(output) if activation else output
    qwen.causal_conv1d_fn = forward
    def restore():
        qwen.causal_conv1d_fn = original
    return restore
