"""Independent fifth-stage Owner, no service integration."""
import sys
sys.path.insert(0,'/tail')
from runner import Owner as PreviousOwner,state,stats,P,json,time,copy,torch,detailed
import long_kernels as lk
from npu_attention import NpuAttention
from full_mlp import FullMlp
from precise_mlp import PreciseMlp
from fused_conv import FusedConv
import ctypes
class Owner(PreviousOwner):
 def __init__(self):
  self.native_attention=None;self.full_mlp=None;self.precise_mlp=None;self.conv=None;self.scan_original=None
  super().__init__()
  try:
   self.base_attention=self.native_originals[0];self.scan_original=lk.scan
   self.native_attention=NpuAttention(fused=True)
   self.full_mlp=FullMlp(self.bank)
   self.precise_mlp=PreciseMlp(self.model,self.engine,self.candidate.table)
   self.conv=FusedConv(self.candidate.table)
   self.fused_scan=ctypes.CDLL('/experiment/libfused_scan.so')
   for name in ('d3_set_sgemm','d3_set_strsm','d3_chunk_scan'):
    src=getattr(lk.scan,name);dst=getattr(self.fused_scan,name);dst.argtypes=src.argtypes;dst.restype=src.restype
   self.fused_scan.d3_set_sgemm(ctypes.cast(lk.blas.scipy_cblas_sgemm64_,ctypes.c_void_p));self.fused_scan.d3_set_strsm(ctypes.cast(lk.blas.scipy_cblas_strsm64_,ctypes.c_void_p))
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('current','baseline','attention','mlp640','scan_fused','conv_fused','combined','combined_residual','qualified')
  super().use('current' if mode=='current' else 'validate_mlp')
  self.full_mlp.enabled=mode in ('mlp640','combined','combined_residual','qualified')
  self.precise_mlp.enabled=mode in ('combined','combined_residual','qualified')
  self.conv.enabled=mode in ('conv_fused','combined','combined_residual')
  lk.scan=self.fused_scan if mode in ('scan_fused','combined','combined_residual') else self.scan_original
  self.native_attention.residual=mode=='combined_residual'
  if mode in ('attention','combined','combined_residual','qualified'):
   def attention(q,k,v,mask,scale):
    if tuple(q.shape)==(1,8,615,256):return self.native_attention(q,k,v,mask,scale)
    return self.base_attention(q,k,v,mask,scale)
   lk.attention=attention
  self.native_calls_before=self.full_mlp.calls;self.precise_before=self.precise_mlp.calls;self.conv_before=self.conv.calls
 def counts(self):
  return {**super().counts(),'full640_calls':self.full_mlp.calls-self.native_calls_before,'full640_errors':list(self.full_mlp.errors),'precise_calls':self.precise_mlp.calls-self.precise_before,'precise_errors':list(self.precise_mlp.errors),'precise_range_fallbacks':self.precise_mlp.range_fallbacks,'fused_conv_calls':self.conv.calls-self.conv_before,'fused_conv_errors':list(self.conv.errors)}
 def close(self):
  try:
   if self.conv:self.conv.close()
   if self.precise_mlp:self.precise_mlp.close()
   if self.full_mlp:self.full_mlp.close()
  finally:
   try:
    if self.native_attention:self.native_attention.close()
   finally:
    if self.scan_original:lk.scan=self.scan_original
    super().close()
