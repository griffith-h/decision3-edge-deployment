"""Switch only our ABI-identical CPU kernels between synchronized model calls."""
import ctypes,os
from pathlib import Path
import lut_ops,attention_prepare,residual,rotary,cpu_attention,chunk_blas2
ROOT=Path('/work')

def clone_types(new,old):
 for name in ('mm_error','mm_create_core','mm_run','mm_destroy','mm_prepare_bf16','mm_copy_prepared','mm_run_prepared_bf16','mm_run_bf16','mm_run_prepared_scatter_bf16','mm_profile_read'):
  if hasattr(old,name) and hasattr(new,name):
   a=getattr(old,name);b=getattr(new,name);b.argtypes=a.argtypes;b.restype=a.restype
 return new

def copied(path,old,names):
 new=ctypes.CDLL(str(path))
 for name in names:
  a=getattr(old,name);b=getattr(new,name);b.argtypes=a.argtypes;b.restype=a.restype
 return new

class Backend:
 def __init__(self,engine):
  self.engine=engine;self.initial_engine=engine.lib;self.initial=[lut_ops.lib,rotary.lib,cpu_attention.lib,chunk_blas2.lib]
  self.profile=getattr(engine,'_d3_profile',False);self.env_tuned=os.environ.get('D3_CPU_TUNE')=='a72';self.banks={}
  ref='/previous/libprofile_conversion.so' if self.profile else '/previous/libconversion_bridge.so';self.reference=clone_types(ctypes.CDLL(ref),engine.lib)
 def get(self,tuned):
  if tuned in self.banks:return self.banks[tuned]
  root=ROOT/'a72' if tuned else ROOT
  lut=copied(root/'liblut_math.so',self.initial[0],('d3_lut','d3_gate_table','d3_norm_lut','d3_add_norm','d3_attention_prepare','d3_attention_gate'))
  rot=copied(root/'librotary.so',self.initial[1],('d3_rope',))
  attn=copied(root/'libcpu_attention.so',self.initial[2],('d3_set_sgemm','d3_attention'));attn.d3_set_sgemm(ctypes.cast(cpu_attention.blas.scipy_cblas_sgemm64_,ctypes.c_void_p))
  path=(root/'libchunk_blas2.so') if tuned else Path('/attn/libchunk_blas2.so')
  chunk=copied(path,self.initial[3],('d3_set_sgemm','d3_set_strsm','d3_chunk_scan'));chunk.d3_set_sgemm(ctypes.cast(chunk_blas2.blas.scipy_cblas_sgemm64_,ctypes.c_void_p));chunk.d3_set_strsm(ctypes.cast(chunk_blas2.blas.scipy_cblas_strsm64_,ctypes.c_void_p))
  bridge=clone_types(ctypes.CDLL(str(root/('libpipeline_profile.so' if self.profile else 'libpipeline_bridge.so'))),self.initial_engine)
  self.banks[tuned]=(lut,rot,attn,chunk,bridge);return self.banks[tuned]
 def use(self,cfg):
  tuned=cfg.get('cpu_tune',self.env_tuned);lut,rot,attn,chunk,bridge=self.get(tuned)
  lut_ops.lib=attention_prepare.lib=residual.lib=lut;rotary.lib=rot;cpu_attention.lib=attn;chunk_blas2.lib=chunk
  self.engine.lib=bridge if cfg.get('pipeline') or tuned else self.reference
 def reset(self):
  lut,rot,attn,chunk=self.initial;lut_ops.lib=attention_prepare.lib=residual.lib=lut;rotary.lib=rot;cpu_attention.lib=attn;chunk_blas2.lib=chunk;self.engine.lib=self.initial_engine
