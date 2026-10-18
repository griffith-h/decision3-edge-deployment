from extended_owner import Owner as PreviousOwner,state,stats,P,json,time,copy,torch,detailed
from npu_attention import NpuAttention
import long_kernels as lk
import ctypes
class Owner(PreviousOwner):
 def __init__(self,grouped=False):
  self.grouped_attention=None;self.original_attention_object=None;self.use_grouped=grouped
  super().__init__();self.original_attention_object=self.native_attention
  try:
   if grouped:self.grouped_attention=NpuAttention(fused=True,grouped=True)
   self.scan_options={}
   for name,path in [('four','libscan_four.so'),('neon','libneon_scan.so')]:
    lib=ctypes.CDLL('/experiment/'+path)
    for symbol in ('d3_set_sgemm','d3_set_strsm','d3_chunk_scan'):
     a=getattr(lk.scan,symbol);b=getattr(lib,symbol);b.argtypes=a.argtypes;b.restype=a.restype
    lib.d3_set_sgemm(ctypes.cast(lk.blas.scipy_cblas_sgemm64_,ctypes.c_void_p));lib.d3_set_strsm(ctypes.cast(lk.blas.scipy_cblas_strsm64_,ctypes.c_void_p));self.scan_options[name]=lib
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('current','baseline','wide','four','neon')
  self.native_attention=self.grouped_attention if self.use_grouped and mode not in ('current','baseline') else self.original_attention_object
  super().use('wide' if mode in ('four','neon') else mode)
  if mode in self.scan_options:lk.scan=self.scan_options[mode]
 def close(self):
  try:
   if self.grouped_attention:self.grouped_attention.close()
  finally:
   if self.original_attention_object:self.native_attention=self.original_attention_object
   super().close()
