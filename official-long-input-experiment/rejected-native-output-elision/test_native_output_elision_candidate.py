"""Real SDK parity and the exact existing finite/range rejection."""
import ctypes,unittest
from pathlib import Path
import torch
from test_native_copy_layout import library

class CheckedNativeTests(unittest.TestCase):
    def lib(self):
        lib=library(Path(__file__).with_name('liblinear_precision_probe.so'))
        lib.nm_output.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];lib.nm_output.restype=ctypes.c_int
        lib.d3_native_run_checked.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.d3_native_run_checked.restype=ctypes.c_int
        return lib
    def test_completed_fp32_is_bitwise_equal_to_old_output_path(self):
        lib=self.lib()
        for rows,n in ((1,32),(3,32),(129,1024),(295,1024),(320,1024),(513,3584),(615,3584)):
            x=((torch.arange(rows*64).reshape(rows,64)%7)/8).bfloat16()
            weights=((torch.arange(n*64).reshape(n,64)%5)/16).half().t().contiguous()
            h=lib.nm_create(64,n,weights.data_ptr(),1);self.assertTrue(h)
            try:
                self.assertEqual(lib.nm_prepare(h,rows,x.data_ptr()),0)
                bf=torch.empty(rows,n,dtype=torch.bfloat16);old=torch.empty(rows,n,dtype=torch.float32);new=torch.empty_like(old)
                self.assertEqual(lib.nm_output(h,rows,bf.data_ptr()),0);self.assertEqual(lib.d3_probe_copy_fp32(h,rows,old.data_ptr()),0)
                self.assertEqual(lib.d3_native_run_checked(h,rows),0);self.assertEqual(lib.d3_probe_copy_fp32(h,rows,new.data_ptr()),0)
                self.assertTrue(torch.equal(old.view(torch.int32),new.view(torch.int32)))
            finally:lib.nm_destroy(h)
    def test_original_large_finite_guard_and_contract_rejections_remain(self):
        lib=self.lib();weights=torch.full((64,32),300.,dtype=torch.float16);h=lib.nm_create(64,32,weights.data_ptr(),1);self.assertTrue(h)
        try:
            self.assertNotEqual(lib.d3_native_run_checked(h,3),0)
            x=torch.full((3,64),300.,dtype=torch.bfloat16);self.assertEqual(lib.nm_prepare(h,3,x.data_ptr()),0)
            bf=torch.empty(3,32,dtype=torch.bfloat16)
            self.assertEqual(lib.nm_output(h,3,bf.data_ptr()),-3);self.assertEqual(lib.d3_native_run_checked(h,3),-3)
            for rows in (0,2,4):self.assertNotEqual(lib.d3_native_run_checked(h,rows),0)
        finally:lib.nm_destroy(h)

if __name__=='__main__':unittest.main()
