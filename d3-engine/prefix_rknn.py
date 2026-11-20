"""Bounded, dual-core RKNN products for a single shared GQA key/value segment."""

import ctypes
from concurrent.futures import wait
from pathlib import Path

import torch
import native_mlp


class RknnSegments:
    def __init__(self, pool):
        self.pool = pool
        self.lib = ctypes.CDLL(str(Path(native_mlp.__file__).with_name("libd3_segment.so")))
        ptr = ctypes.c_void_p
        self.lib.dm_create.argtypes = [ctypes.c_int] * 4
        self.lib.dm_create.restype = ptr
        self.lib.dm_destroy.argtypes = [ptr]
        self.lib.dm_destroy.restype = None
        self.lib.dm_error.restype = ctypes.c_char_p
        self.lib.d3_segment.argtypes = [ptr] * 7 + [ctypes.c_int] * 4 + [ctypes.c_float]
        self.lib.d3_segment.restype = ctypes.c_int
        self.handles = []
        self.width = 0

    @staticmethod
    def long_keys(k):
        return k.shape[:2] == (1, 2) and k.shape[2] >= 2048 and k.shape[-1] == 256 and k.dtype == torch.bfloat16

    @classmethod
    def supports(cls, q, k):
        return cls.long_keys(k) and q.shape[2] >= 16 and q.shape[1] % 2 == 0 and q.dtype == torch.bfloat16

    def close(self):
        for handle in self.handles:
            self.lib.dm_destroy(handle)
        self.handles.clear()
        self.width = 0

    def prepare(self, width):
        if self.width == width:
            return
        self.close()
        try:
            for core in (1, 2):
                for k, n in ((256, width), (width, 256)):
                    h = self.lib.dm_create(128, k, n, core)
                    if not h:
                        raise RuntimeError(f"RKNN segment allocation: {self.lib.dm_error()!r}")
                    self.handles.append(h)
            self.width = width
        except BaseException:
            self.close()
            raise

    def __call__(self, q, k, v, scale):
        batch, heads, length, dim = q.shape
        keys = k.shape[2]
        blocks = (keys + 8191) // 8192
        width = ((keys + blocks - 1) // blocks + 127) // 128 * 128
        self.prepare(width)
        groups = heads // 2
        packed = q.reshape(batch, 2, groups, length, dim).permute(1, 0, 2, 3, 4).contiguous()
        packed = packed.view(2, batch * groups * length, dim)
        k, v = k[0].contiguous(), v[0].contiguous()
        out = torch.empty(packed.shape, dtype=torch.float32)
        lse = torch.empty(packed.shape[:-1], dtype=torch.float32)

        def head(i):
            rc = self.lib.d3_segment(*self.handles[2*i:2*i+2], packed[i].data_ptr(), k[i].data_ptr(),
                                     v[i].data_ptr(), out[i].data_ptr(), lse[i].data_ptr(),
                                     packed.shape[1], keys, dim, width, scale)
            if rc not in (0, -2):
                raise RuntimeError(f"RKNN attention: {self.lib.dm_error()!r} ({rc})")
            return rc

        jobs = []
        try:
            for i in range(2):
                jobs.append(self.pool.submit(head, i))
        finally:
            if jobs:
                wait(jobs)
        codes = [job.result() for job in jobs]
        if -2 in codes:
            return None
        out = out.view(2, batch, groups, length, dim).permute(1, 0, 2, 3, 4)
        lse = lse.view(2, batch, groups, length).permute(1, 0, 2, 3)
        return out.reshape(batch, heads, length, dim), lse.reshape(batch, heads, length)
