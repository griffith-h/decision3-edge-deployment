"""Experimental FP16 RKNN Linear adapter; original attention stays in PyTorch."""
import ctypes
import time
from pathlib import Path
from types import MethodType
import torch


class Engine:
    def __init__(self):
        self.lib = ctypes.CDLL(str(Path(__file__).with_name('libmatmul_bridge.so')))
        self.lib.mm_error.restype = ctypes.c_char_p
        self.lib.mm_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.mm_create.restype = ctypes.c_void_p
        self.lib.mm_run.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.mm_run.restype = ctypes.c_int
        self.lib.mm_destroy.argtypes = [ctypes.c_void_p]
        self.handles = []
        self.calls = self.cpu_fallbacks = 0
        self.seconds = self.create_seconds = 0.0

    def error(self):
        return RuntimeError(self.lib.mm_error().decode())

    def create(self, weight):
        # PyTorch weight [N,K] -> official normal B layout [K,N], then native-pack once.
        start = time.perf_counter()
        half = weight.detach().to(torch.float16).t().contiguous()
        if not torch.isfinite(half).all():
            raise ValueError('FP16 weight overflow')
        handle = self.lib.mm_create(weight.shape[1], weight.shape[0], half.data_ptr())
        if not handle:
            raise self.error()
        self.handles.append(handle)
        self.create_seconds += time.perf_counter() - start
        return handle

    def run(self, handle, inputs, out_features):
        start = time.perf_counter()
        flat = inputs.reshape(-1, inputs.shape[-1]).to(torch.float16).contiguous()
        if not torch.isfinite(flat).all():
            raise ValueError('FP16 activation overflow')
        output = torch.empty((flat.shape[0], out_features), dtype=torch.float32)
        if self.lib.mm_run(handle, flat.shape[0], flat.data_ptr(), output.data_ptr()):
            raise self.error()
        self.calls += 1
        self.seconds += time.perf_counter() - start
        return output.reshape(*inputs.shape[:-1], out_features)

    def close(self):
        for handle in self.handles:
            self.lib.mm_destroy(handle)
        self.handles.clear()


def install(model, engine, selection='all'):
    """Replace only CPU BF16 text Linear.forward, after existing CPU optimization.

    Unsupported large requests use the saved CPU forward. No runtime error is
    silently hidden. Handles/weights live for the whole benchmark, not each call.
    """
    chosen = []
    for name, module in model.backbone.named_modules():
        if 'language_model.layers.' not in name or not isinstance(module, torch.nn.Linear):
            continue
        if selection == 'mlp' and '.mlp.' not in name:
            continue
        k, n = module.in_features, module.out_features
        if k > 10240 or k % 16 or n % 8 or k * n < 262144:
            continue
        module._rknn_cpu_forward = module.forward
        module._rknn_handle = None
        def forward(self, inputs):
            rows = inputs.numel() // inputs.shape[-1]
            if inputs.device.type != 'cpu' or inputs.dtype != torch.bfloat16 or rows > 512:
                engine.cpu_fallbacks += 1
                return self._rknn_cpu_forward(inputs)
            if self._rknn_handle is None:
                self._rknn_handle = engine.create(self.weight)
            output = engine.run(self._rknn_handle, inputs, self.out_features)
            if self.bias is not None:
                output += self.bias.float()
            return output.to(inputs.dtype)
        module.forward = MethodType(forward, module)
        chosen.append({'name': name, 'K': k, 'N': n})
    return chosen


def uninstall(model):
    # Restore CPU forwards before freeing NPU buffers, so retained model objects
    # cannot reference released native handles.
    for module in model.backbone.modules():
        if hasattr(module, '_rknn_cpu_forward'):
            module.forward = module._rknn_cpu_forward
            del module._rknn_cpu_forward
            del module._rknn_handle
