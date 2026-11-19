"""CPU kernels bound to one checkpoint's StateOnce instance."""

import ctypes
import time
from pathlib import Path

import numpy
import torch


def pointer(tensor):
    return tensor.data_ptr() if tensor is not None else None


class StateKernels:
    def __init__(self, original, counts, library):
        self.original = original
        self.counts = counts
        libs = Path(numpy.__file__).resolve().parent.parent / "numpy.libs"
        (blas,) = libs.glob("libscipy_openblas64_*.so")
        self.blas = ctypes.CDLL(str(blas), mode=ctypes.RTLD_GLOBAL)
        native = ctypes.CDLL(str(library))
        self.scan = native.d3_scan_state
        self.scan.argtypes = [ctypes.c_void_p] * 8 + [ctypes.c_int] * 4
        self.prepare = native.d3_prepare_scan
        self.prepare.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 6
        self.conv = native.d3_conv_btc
        self.conv.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 3

    def __getattr__(self, key):
        return getattr(self.original, key)

    def causal_conv1d_fn(self, x, weight, bias=None, activation=None, **kwargs):
        if (torch.is_grad_enabled() or x.ndim != 3 or x.device.type != "cpu" or x.dtype != torch.bfloat16
                or not all(x.shape) or x.shape[1] % 4 or weight.shape != (x.shape[1], 4)
                or weight.dtype != x.dtype or weight.device != x.device
                or (bias is not None and (bias.shape != (x.shape[1],) or bias.dtype != x.dtype or bias.device != x.device))
                or activation not in (None, "silu", "swish") or kwargs):
            self.counts["conv_fallback"] += 1
            return self.original.causal_conv1d_fn(x, weight, bias, activation=activation, **kwargs)
        # Question continuations concatenate their convolution history in channel-major order.
        start = time.perf_counter()
        rows = x.transpose(1, 2).contiguous()
        out = torch.empty_like(rows)
        w = weight.contiguous()
        b = bias.contiguous() if bias is not None else None
        if self.conv(*(pointer(t) for t in (rows, w, b, out)), *rows.shape):
            raise RuntimeError("native causal convolution failed")
        self.counts["conv"] += 1
        out = out.transpose(1, 2)
        out = torch.nn.functional.silu(out) if activation else out
        self.counts["conv_seconds"] += time.perf_counter() - start
        return out

    def torch_chunk_gated_delta_rule(self, query, key, value, g, beta, chunk_size=64, initial_state=None,
                                    output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        supported = (
            not torch.is_grad_enabled() and chunk_size == 64 and not kwargs
            and all(x.device.type == "cpu" for x in (query, key, value, g, beta))
            and all(x.dtype == torch.bfloat16 and x.ndim == 4 and x.stride(-1) == 1 for x in (query, key, value))
            and all(x.dtype in (torch.float32, torch.bfloat16) for x in (g, beta))
            and query.shape == key.shape and query.shape[:3] == value.shape[:3] == g.shape == beta.shape
            and all(query.shape) and all(0 < d <= 128 and d % 4 == 0 for d in (query.shape[-1], value.shape[-1]))
        )
        if supported:
            batch, length, heads, kd = query.shape
            vd = value.shape[-1]
            state_shape = (batch, heads, kd, vd)
            supported = (initial_state is None or (initial_state.device.type == "cpu"
                         and initial_state.dtype == torch.float32 and tuple(initial_state.shape) == state_shape))
        if not supported:
            self.counts["scan_fallback"] += 1
            return self.original.torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, chunk_size=chunk_size, initial_state=initial_state,
                output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs)

        start = time.perf_counter()
        q = torch.empty((batch, heads, length, kd), dtype=torch.float32)
        k = torch.empty_like(q)
        v = torch.empty((batch, heads, length, vd), dtype=torch.float32)
        strides = (ctypes.c_int64 * 9)(*(s for x in (query, key, value) for s in x.stride()[:3]))
        if self.prepare(*(pointer(t) for t in (query, key, value, q, k, v)), ctypes.addressof(strides),
                        batch, length, heads, kd, vd, int(use_qk_l2norm_in_kernel)):
            raise RuntimeError("native scan preparation failed")
        decay = g.transpose(1, 2).float().contiguous().exp()
        beta = beta.transpose(1, 2).float().contiguous()
        initial = initial_state.contiguous() if initial_state is not None else None
        final = torch.empty(state_shape, dtype=torch.float32) if output_final_state else None
        out = torch.empty(v.shape, dtype=torch.bfloat16)
        if self.scan(*(pointer(t) for t in (q, k, v, decay, beta, initial, final, out)), batch*heads, length, kd, vd):
            raise RuntimeError("native stateful scan failed")
        self.counts["scan_continue" if initial is not None else "scan_zero"] += 1
        out = out.transpose(1, 2).contiguous()
        self.counts["scan_seconds"] += time.perf_counter() - start
        return out, final
