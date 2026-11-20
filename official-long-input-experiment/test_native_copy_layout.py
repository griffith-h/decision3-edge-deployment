"""Actual public-SDK FP32 layout copy, partial rows and early rejections."""
import ctypes,unittest
from pathlib import Path
import torch
import torch.nn.functional as f

def library(path):
    lib=ctypes.CDLL(str(path))
    lib.nm_create.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int];lib.nm_create.restype=ctypes.c_void_p
    lib.nm_destroy.argtypes=[ctypes.c_void_p]
    lib.nm_prepare.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];lib.nm_prepare.restype=ctypes.c_int
    lib.nm_raw.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.nm_raw.restype=ctypes.c_int
    lib.d3_probe_copy_fp32.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];lib.d3_probe_copy_fp32.restype=ctypes.c_int
    return lib

class NativeCopyTests(unittest.TestCase):
    def test_exact_layout_partial_rows_and_no_write_past_rows(self):
        lib=library(Path(__file__).with_name('liblinear_precision_probe.so'))
        for rows,n in ((1,32),(3,32),(129,1024),(513,3584),(615,3584)):
            x=((torch.arange(rows*64).reshape(rows,64)%7)/8).bfloat16()
            weight=((torch.arange(n*64).reshape(n,64)%5)/16).half();packed=weight.t().contiguous()
            handle=lib.nm_create(64,n,packed.data_ptr(),1);self.assertTrue(handle)
            try:
                self.assertEqual(lib.nm_prepare(handle,rows,x.data_ptr()),0)
                self.assertEqual(lib.nm_raw(handle,rows),0)
                target=torch.full((rows+1,n),77.,dtype=torch.float32)
                self.assertEqual(lib.d3_probe_copy_fp32(handle,rows,target.data_ptr()),0)
                self.assertTrue(torch.equal(target[:rows].view(torch.int32),f.linear(x.float(),weight.float()).view(torch.int32)))
                self.assertTrue(bool((target[rows:]==77).all()))
            finally:lib.nm_destroy(handle)
    def test_unprepared_or_wrong_rows_reject_before_output_mutation(self):
        lib=library(Path(__file__).with_name('liblinear_precision_probe.so'))
        packed=torch.ones((64,32),dtype=torch.float16);handle=lib.nm_create(64,32,packed.data_ptr(),1);self.assertTrue(handle)
        target=torch.full((4,32),77.,dtype=torch.float32)
        try:
            self.assertNotEqual(lib.d3_probe_copy_fp32(handle,3,target.data_ptr()),0)
            x=torch.ones((3,64),dtype=torch.bfloat16)
            self.assertEqual(lib.nm_prepare(handle,3,x.data_ptr()),0)
            self.assertEqual(lib.nm_raw(handle,3),0)
            for rows in (0,2,4):
                self.assertNotEqual(lib.d3_probe_copy_fp32(handle,rows,target.data_ptr()),0)
                self.assertTrue(bool((target==77).all()))
        finally:lib.nm_destroy(handle)

if __name__=='__main__':unittest.main()
