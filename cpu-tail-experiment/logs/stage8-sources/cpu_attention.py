import ctypes,glob,torch
from pathlib import Path
from lut_ops import native_path
lib=ctypes.CDLL(str(native_path('libcpu_attention.so')))
blas=ctypes.CDLL(glob.glob('/usr/local/lib/python3.12/site-packages/numpy.libs/libscipy_openblas64_*.so')[0])
lib.d3_set_sgemm.argtypes=[ctypes.c_void_p];lib.d3_set_sgemm(ctypes.cast(blas.scipy_cblas_sgemm64_,ctypes.c_void_p))
lib.d3_attention.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*5+[ctypes.c_float,ctypes.c_int];lib.d3_attention.restype=ctypes.c_int

def attention(query,key,value,mask,scale,mode=1):
 b,h,t,d=query.shape;strides=(ctypes.c_int64*9)(*[s for x in (query,key,value) for s in x.stride()[:3]])
 out=torch.empty((b,t,h,d),dtype=query.dtype)
 rc=lib.d3_attention(*(x.data_ptr() for x in (query,key,value)),mask.data_ptr() if mask is not None else None,out.data_ptr(),ctypes.addressof(strides),b,h,key.shape[1],t,d,scale,mode)
 if rc:raise ValueError('CPU attention rejects nonfinite input/output or allocation failure')
 return out

def install(mode=1):
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 interface=qwen.ALL_ATTENTION_FUNCTIONS;original=interface['sdpa'];counts={'optimized':0,'fallback':0,'range_fallback':0}
 def forward(module,query,key,value,attention_mask,*args,**kwargs):
  good=(not torch.is_grad_enabled() and not module.training and type(module) is qwen.Qwen3_5Attention and not args and
   all(x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.ndim==4 and x.stride(-1)==1 and all(s>=0 for s in x.stride()) for x in (query,key,value)) and
   query.shape[0]>0 and 0<query.shape[2]<=512 and 0<query.shape[3]<=256 and key.shape==value.shape and query.shape[0]==key.shape[0] and query.shape[2:]==key.shape[2:] and key.shape[1]>0 and query.shape[1]%key.shape[1]==0 and
   attention_mask is not None and attention_mask.dtype==torch.bool and attention_mask.device.type=='cpu' and attention_mask.shape==(query.shape[0],1,1,query.shape[2]) and bool(attention_mask.any(dim=-1).all()) and
   not kwargs.get('use_cache',False) and kwargs.get('dropout',0)==0 and not kwargs.get('is_causal',False) and
   all(k in ('dropout','scaling','position_ids','use_cache','is_causal') or v is None for k,v in kwargs.items()))
  def fallback():counts['fallback']+=1;return original(module,query,key,value,attention_mask,*args,**kwargs)
  if not good:return fallback()
  try:out=attention(query,key,value,attention_mask.contiguous(),kwargs.get('scaling',module.scaling),mode)
  except ValueError:counts['range_fallback']+=1;return fallback()
  counts['optimized']+=1;return out,None
 interface.register('sdpa',forward)
 def restore():interface.register('sdpa',original)
 return restore,counts
