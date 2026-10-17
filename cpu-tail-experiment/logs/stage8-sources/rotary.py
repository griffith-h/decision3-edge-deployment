import ctypes,torch
from pathlib import Path
from lut_ops import good,native_path
lib=ctypes.CDLL(str(native_path('librotary.so')))
lib.d3_rope.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*6;lib.d3_rope.restype=ctypes.c_int

def rotary(q,k,cos,sin):
 qo=torch.empty(q.shape,dtype=q.dtype);ko=torch.empty(k.shape,dtype=k.dtype);strides=(ctypes.c_int64*6)(*q.stride()[:3],*k.stride()[:3]);cos=cos.contiguous();sin=sin.contiguous()
 rc=lib.d3_rope(*(x.data_ptr() for x in (q,k,cos,sin,qo,ko)),ctypes.addressof(strides),q.shape[0],q.shape[1],k.shape[1],q.shape[2],q.shape[3],cos.shape[-1])
 if rc:raise RuntimeError('Validated rotary rejected')
 return qo,ko

def install():
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 old=qwen.apply_rotary_pos_emb;counts={'optimized':0,'fallback':0}
 def forward(q,k,cos,sin,unsqueeze_dim=1):
  if (unsqueeze_dim!=1 or not all(good(x) for x in (q,k,cos,sin)) or q.ndim!=4 or k.ndim!=4 or q.stride(-1)!=1 or k.stride(-1)!=1 or
      q.shape[0]!=k.shape[0] or q.shape[2:]!=k.shape[2:] or cos.shape!=sin.shape or cos.shape[:2]!=(q.shape[0],q.shape[2]) or cos.ndim!=3 or cos.shape[-1]%8 or not 0<cos.shape[-1]<=q.shape[-1]):
   counts['fallback']+=1;return old(q,k,cos,sin,unsqueeze_dim)
  counts['optimized']+=1;return rotary(q,k,cos,sin)
 qwen.apply_rotary_pos_emb=forward
 def restore():qwen.apply_rotary_pos_emb=old
 return restore,counts
