"""Bounded long-sequence extensions of our own FP32 CPU kernels.

Only 512<T<=2048 is intercepted. Short, cached, causal or unknown contracts are
delegated verbatim. A failed native result is discarded as a whole.
"""
import ctypes
import glob
import time
from pathlib import Path
import torch

P = Path(__file__).resolve().parent
blas = ctypes.CDLL(glob.glob('/usr/local/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_*.so')[0])
blas.scipy_openblas_set_num_threads64_.argtypes = [ctypes.c_int64]
blas.scipy_openblas_set_num_threads64_(1)
att = ctypes.CDLL(str(P / 'liblong_attention.so'))
scan = ctypes.CDLL(str(P / 'liblong_scan.so'))
for lib in (att, scan):
    lib.d3_set_sgemm.argtypes = [ctypes.c_void_p]
    lib.d3_set_sgemm(ctypes.cast(blas.scipy_cblas_sgemm64_, ctypes.c_void_p))
scan.d3_set_strsm.argtypes = [ctypes.c_void_p]
scan.d3_set_strsm(ctypes.cast(blas.scipy_cblas_strsm64_, ctypes.c_void_p))
att.d3_attention.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*5 + [ctypes.c_float, ctypes.c_int]
att.d3_attention.restype = ctypes.c_int
scan.d3_chunk_scan.argtypes = [ctypes.c_void_p]*6 + [ctypes.c_int]*6
scan.d3_chunk_scan.restype = ctypes.c_int

def attention(q, k, v, mask, scale):
    b, h, t, d = q.shape
    out = torch.empty((b, t, h, d), dtype=q.dtype)
    strides = (ctypes.c_int64*9)(*[s for x in (q, k, v) for s in x.stride()[:3]])
    rc = att.d3_attention(*(x.data_ptr() for x in (q, k, v)), mask.data_ptr(), out.data_ptr(), ctypes.addressof(strides), b, h, k.shape[1], t, d, scale, 1)
    if rc:
        raise ValueError('long attention rejected: '+str(rc))
    return out

def scan_prepared(q, k, v, decay, beta):
    out = torch.empty(v.shape, dtype=torch.bfloat16)
    rc = scan.d3_chunk_scan(*(x.data_ptr() for x in (q, k, v, decay, beta, out)), q.shape[0]*q.shape[1], q.shape[2], q.shape[3], v.shape[3], 32, 64)
    if rc or not bool(torch.isfinite(out).all()):
        raise ValueError('long scan rejected: '+str(rc))
    return out

class Kernels:
    def __init__(self):
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        self.qwen = qwen
        self.original_attention = qwen.ALL_ATTENTION_FUNCTIONS['sdpa']
        self.original_scan = qwen.torch_chunk_gated_delta_rule
        self.enabled = set()
        self.disabled = set()
        self.profile = False
        self.closed = False
        self.reset()
        qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa', self.attention)
        qwen.torch_chunk_gated_delta_rule = self.scan

    def reset(self):
        self.counts = {k: {'optimized': 0, 'bypass': 0, 'guard_fallback': 0, 'native_fallback': 0, 'seconds': 0., 'lengths': {}, 'errors': []} for k in ('attention', 'scan')}

    def call(self, kind, good, length, native, fallback):
        c = self.counts[kind]
        start = time.perf_counter() if self.profile else None
        try:
            if kind not in self.enabled or length <= 512:
                c['bypass'] += 1
                return fallback()
            if not good or kind in self.disabled:
                c['guard_fallback'] += 1
                return fallback()
            try:
                out = native()
            except (ValueError, RuntimeError) as exc:
                c['native_fallback'] += 1
                c['errors'].append(str(exc))
                self.disabled.add(kind)
                return fallback()
            c['optimized'] += 1
            c['lengths'][str(length)] = c['lengths'].get(str(length), 0)+1
            return out
        finally:
            if start is not None:
                c['seconds'] += time.perf_counter()-start

    def attention(self, module, query, key, value, attention_mask, *args, **kwargs):
        xs = (query, key, value)
        length = query.shape[2] if query.ndim == 4 else 0
        good = (
            not torch.is_grad_enabled() and not module.training and type(module) is self.qwen.Qwen3_5Attention and not args and
            all(x.device.type == 'cpu' and x.dtype == torch.bfloat16 and x.layout == torch.strided and x.ndim == 4 and x.stride(-1) == 1 and all(s >= 0 for s in x.stride()) for x in xs) and
            512 < length <= 2048 and 0 < query.shape[0] <= 8 and 0 < query.shape[1] <= 64 and query.shape[0]*query.shape[1] <= 128 and 0 < query.shape[-1] <= 256 and
            key.shape == value.shape and query.shape[0] == key.shape[0] and query.shape[2:] == key.shape[2:] and key.shape[1] > 0 and query.shape[1] % key.shape[1] == 0 and
            attention_mask is not None and attention_mask.device.type == 'cpu' and attention_mask.dtype == torch.bool and attention_mask.shape == (query.shape[0], 1, 1, length) and bool(attention_mask.any(dim=-1).all()) and
            not kwargs.get('use_cache', False) and kwargs.get('dropout', 0) == 0 and not kwargs.get('is_causal', False) and
            all(k in ('dropout', 'scaling', 'position_ids', 'use_cache', 'is_causal') or v is None for k, v in kwargs.items()))
        return self.call('attention', good, length,
                         lambda: (attention(query, key, value, attention_mask.contiguous(), kwargs.get('scaling', module.scaling)), None),
                         lambda: self.original_attention(module, query, key, value, attention_mask, *args, **kwargs))

    def scan(self, query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        xs = (query, key, value, g, beta)
        length = query.shape[1] if query.ndim == 4 else 0
        good = (
            not torch.is_grad_enabled() and all(x.device.type == 'cpu' and x.layout == torch.strided for x in xs) and
            query.dtype == key.dtype == value.dtype == torch.bfloat16 and g.dtype in (torch.bfloat16, torch.float32) and beta.dtype in (torch.bfloat16, torch.float32) and
            query.ndim == key.ndim == value.ndim == 4 and g.ndim == beta.ndim == 3 and query.shape == key.shape and query.shape[:3] == value.shape[:3] and g.shape == beta.shape == query.shape[:3] and
            512 < length <= 2048 and 0 < query.shape[0] <= 8 and query.shape[2] > 0 and query.shape[0]*query.shape[2] <= 128 and
            0 < query.shape[-1] <= 128 and query.shape[-1] % 4 == 0 and 0 < value.shape[-1] <= 128 and value.shape[-1] % 4 == 0 and
            all(x.stride(-1) == 1 and all(s >= 0 for s in x.stride()) for x in (query, key, value)) and
            initial_state is None and not output_final_state and type(chunk_size) is int and chunk_size > 0 and
            all(v is None or (k == 'use_cache' and v is False) for k, v in kwargs.items()))
        def native():
            from prepared_scan import prepare
            # Domains produced by the unmodified model; other callers go to the
            # original path rather than introducing a new numerical contract.
            if not all(bool(torch.isfinite(x).all()) for x in xs) or bool((g > 0).any()) or bool(((beta < 0) | (beta > 1)).any()):
                raise ValueError('nonfinite or non-model scan domain')
            q, k, v = prepare(query, key, value, use_qk_l2norm_in_kernel)
            b = beta.transpose(1, 2).float().contiguous()
            decay = g.transpose(1, 2).float().contiguous().exp()
            if not all(bool(torch.isfinite(x).all()) for x in (q, k, v, decay)):
                raise ValueError('nonfinite prepared scan')
            return scan_prepared(q, k, v, decay, b).transpose(1, 2).contiguous(), None
        return self.call('scan', good, length, native,
                         lambda: self.original_scan(query, key, value, g, beta, chunk_size=chunk_size, initial_state=initial_state, output_final_state=output_final_state, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kwargs))

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa', self.original_attention)
        self.qwen.torch_chunk_gated_delta_rule = self.original_scan
