"""Public-SDK normal FP32 copies and rejection before output mutation."""
import ctypes,unittest
from pathlib import Path
import torch
import torch.nn.functional as f

class ProjectionReaderTests(unittest.TestCase):
    def test_actual_fp32_layout_and_half_or_mismatched_context_rejection(self):
        lib=ctypes.CDLL(str(Path(__file__).with_name('libprojection_read_probe.so')))
        lib.mm_create_core.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int];lib.mm_create_core.restype=ctypes.c_void_p
        lib.mm_set_output_precision.argtypes=[ctypes.c_int];lib.mm_set_output_precision.restype=ctypes.c_int
        lib.mm_destroy.argtypes=[ctypes.c_void_p]
        lib.mm_run_bf16.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];lib.mm_run_bf16.restype=ctypes.c_int
        lib.d3_projection_read_fp32.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.d3_projection_read_fp32.restype=ctypes.c_int
        for precise,rows in ((True,3),(True,129),(False,3)):
            weights=((torch.arange(32*64).reshape(32,64)%5).half()/16);packed=weights.t().contiguous()
            x=(torch.arange(rows*64).reshape(rows,64)%7).bfloat16()/8
            previous=lib.mm_set_output_precision(int(precise))
            try:handle=lib.mm_create_core(64,32,packed.data_ptr(),1)
            finally:lib.mm_set_output_precision(previous)
            self.assertTrue(handle)
            try:
                bf16=torch.empty((rows,32),dtype=torch.bfloat16);raw=torch.full((rows,32),77.)
                self.assertNotEqual(lib.d3_projection_read_fp32(handle,rows,64,32,raw.data_ptr()),0)
                self.assertTrue(bool((raw==77).all()))
                self.assertEqual(lib.mm_run_bf16(handle,rows,x.data_ptr(),None,bf16.data_ptr()),0)
                result=lib.d3_projection_read_fp32(handle,rows,64,32,raw.data_ptr())
                if precise:
                    self.assertEqual(result,0);self.assertTrue(torch.equal(raw,f.linear(x.float(),weights.float())))
                    self.assertTrue(torch.equal(bf16,raw.bfloat16()))
                else:self.assertNotEqual(result,0);self.assertTrue(bool((raw==77).all()))
                raw.fill_(77)
                for b,k,n in ((rows,32,32),(rows,64,16),(rows+1,64,32),(513,64,32)):
                    self.assertNotEqual(lib.d3_projection_read_fp32(handle,b,k,n,raw.data_ptr()),0)
                    self.assertTrue(bool((raw==77).all()))
            finally:lib.mm_destroy(handle)

if __name__=='__main__':unittest.main()
