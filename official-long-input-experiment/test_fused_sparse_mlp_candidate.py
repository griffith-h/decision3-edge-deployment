import ctypes,unittest
from pathlib import Path
import torch
import torch.nn.functional as f

class CorrectedProductTests(unittest.TestCase):
    def test_actual_native_slices_keep_original_bf16_product_and_padding(self):
        reader=ctypes.CDLL(str(Path(__file__).with_name('liblinear_precision_probe.so')))
        lib=ctypes.CDLL('/native/libnative_split.so')
        lib.nm_create.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int];lib.nm_create.restype=ctypes.c_void_p
        lib.nm_destroy.argtypes=[ctypes.c_void_p]
        lib.nm_output.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];lib.nm_output.restype=ctypes.c_int
        reader.d3_native_product_bf16.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]*2;reader.d3_native_product_bf16.restype=ctypes.c_int
        table=f.silu(torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)).contiguous()
        torch.manual_seed(12);gate=torch.randn(615,64,dtype=torch.bfloat16);up=torch.randn_like(gate)
        expected=f.silu(gate)*up;product=torch.full_like(gate,77.)
        weights=torch.zeros(32,64,dtype=torch.float16);weights[:,:32]=torch.eye(32,dtype=torch.float16);packed=weights.t().contiguous()
        for core,(start,rows) in enumerate(((0,320),(320,295)),1):
            handle=lib.nm_create(64,32,packed.data_ptr(),core);self.assertTrue(handle)
            try:
                self.assertEqual(reader.d3_native_product_bf16(handle,gate.data_ptr(),up.data_ptr(),start,rows,615,64,table.data_ptr(),product.data_ptr()),0)
                self.assertTrue(torch.equal(product[start:start+rows],expected[start:start+rows]))
                output=torch.empty(rows,32,dtype=torch.bfloat16);self.assertEqual(lib.nm_output(handle,rows,output.data_ptr()),0)
                target=f.linear(expected[start:start+rows].half().float(),weights.float()).bfloat16()
                self.assertTrue(torch.equal(output,target))
                before=product.clone()
                self.assertNotEqual(reader.d3_native_product_bf16(handle,gate.data_ptr(),up.data_ptr(),321,295,615,64,table.data_ptr(),product.data_ptr()),0)
                self.assertTrue(torch.equal(product,before))
                large=torch.full_like(gate,65536.);ones=torch.ones_like(up)
                self.assertEqual(reader.d3_native_product_bf16(handle,large.data_ptr(),ones.data_ptr(),start,rows,615,64,table.data_ptr(),product.data_ptr()),-2)
            finally:lib.nm_destroy(handle)

if __name__=='__main__':unittest.main()
