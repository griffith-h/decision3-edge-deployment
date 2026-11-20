"""Bounded FP32 full attention for 2048<T<=8192; all keys remain visible."""
import ctypes,math,hashlib,json
from pathlib import Path
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
import long_kernels as lk

root=Path(__file__).resolve().parent
build=json.loads((root/'attention-build.json').read_text())
for name,digest in build['sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Extended attention source/binary build identity mismatch: '+name)
lib=ctypes.CDLL(str(root/'liblong_attention_extended.so'))
lib.d3_set_sgemm.argtypes=[ctypes.c_void_p]
lib.d3_set_sgemm(ctypes.cast(lk.blas.scipy_cblas_sgemm64_,ctypes.c_void_p))
lib.d3_attention_tiled.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*5+[ctypes.c_float]+[ctypes.c_int]*3
lib.d3_attention_tiled.restype=ctypes.c_int

def supported(q,k,v,mask,scale):
    if any(not isinstance(x,torch.Tensor) for x in (q,k,v,mask)):return False
    if any(x.ndim!=4 for x in (q,k,v)):return False
    b,h,t,d=q.shape
    return (not torch.is_grad_enabled() and 0<b<=8 and 0<h<=64 and b*h<=128
        and 2048<t<=8192 and 0<d<=256 and k.shape==v.shape and k.shape[0]==b
        and k.shape[2:]==(t,d) and k.shape[1]>0 and h%k.shape[1]==0
        and (b*(h+2*k.shape[1])*t*d*4)<=128*1024*1024
        and all(x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.layout==torch.strided
                and x.stride(-1)==1 and all(s>=0 for s in x.stride()) for x in (q,k,v))
        and mask.device.type=='cpu' and mask.dtype==torch.bool and tuple(mask.shape)==(b,1,1,t)
        and bool(mask.any(dim=-1).all()) and isinstance(scale,(int,float)) and math.isfinite(scale))

def kernel(q,k,v,mask,scale):
    if not supported(q,k,v,mask,scale):raise ValueError('Unsupported extended attention contract')
    b,h,t,d=q.shape;out=torch.empty((b,t,h,d),dtype=torch.bfloat16)
    strides=(ctypes.c_int64*9)(*[s for x in (q,k,v) for s in x.stride()[:3]])
    mask=mask.contiguous()
    rc=lib.d3_attention_tiled(*(x.data_ptr() for x in (q,k,v)),mask.data_ptr(),out.data_ptr(),
        ctypes.addressof(strides),b,h,k.shape[1],t,d,scale,64,3,1)
    if rc or not bool(torch.isfinite(out).all()):raise ValueError('Extended attention rejected '+str(rc))
    return out

class LongAttention:
    def __init__(self):
        self.registry=qwen.ALL_ATTENTION_FUNCTIONS;self.previous=self.registry['sdpa']
        self.closed=False;self.disabled=False
        self.counts={'optimized':0,'bypass':0,'fallback':0,'errors':[],'lengths':{}}
        self.registry.register('sdpa',self.forward)
    def forward(self,module,query,key,value,attention_mask,*args,**kwargs):
        scale=kwargs.get('scaling',getattr(module,'scaling',None))
        good=(not self.closed and not self.disabled and type(module) is qwen.Qwen3_5Attention
            and not module.training and not args and kwargs.get('dropout',0)==0
            and not kwargs.get('use_cache',False) and not kwargs.get('is_causal',False)
            and all(k in ('dropout','scaling','position_ids','use_cache','is_causal') or v is None for k,v in kwargs.items())
            and supported(query,key,value,attention_mask,scale))
        if good:
            try:result=kernel(query,key,value,attention_mask,scale)
            except (ValueError,RuntimeError) as exc:
                self.counts['fallback']+=1;self.counts['errors'].append(str(exc));self.disabled=True
            else:
                self.counts['optimized']+=1;t=str(query.shape[2]);self.counts['lengths'][t]=self.counts['lengths'].get(t,0)+1
                return result,None
        else:self.counts['bypass']+=1
        return self.previous(module,query,key,value,attention_mask,*args,**kwargs)
    def close(self):
        if self.closed:return
        self.closed=True
        if self.registry['sdpa']==self.forward:self.registry.register('sdpa',self.previous)

def install(owner):
    owner.long_attention=LongAttention()
    return owner.long_attention.close
