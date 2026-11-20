"""Bounded full-key dynamic NPU attention; FP32 softmax, scaled FP16 probability path."""
import ctypes,hashlib,json,math
from concurrent.futures import wait
from pathlib import Path
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

root=Path(__file__).resolve().parent
build=json.loads((root/'npu-attention-build.json').read_text())
for name,digest in build['sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Long NPU attention build identity mismatch: '+name)
lib=ctypes.CDLL(str(root/'liblong_npu_attention.so'))
lib.la_error.restype=ctypes.c_char_p
lib.la_create.argtypes=[ctypes.c_int]*4;lib.la_create.restype=ctypes.c_void_p
lib.la_destroy.argtypes=[ctypes.c_void_p];lib.la_destroy.restype=None
lib.la_set_b.argtypes=[ctypes.c_void_p]*2;lib.la_set_b.restype=ctypes.c_int
lib.la_run.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int];lib.la_run.restype=ctypes.c_int
lib.la_long_worker.argtypes=[ctypes.c_void_p]*8+[ctypes.c_float]+[ctypes.c_int]*7
lib.la_long_worker.restype=ctypes.c_int

def supported(q,k,v,mask,scale,min_length=2048):
    if any(not isinstance(x,torch.Tensor) for x in (q,k,v,mask)):return False
    if any(x.ndim!=4 for x in (q,k,v)):return False
    b,h,t,d=q.shape
    return (not torch.is_grad_enabled() and 0<b<=8 and 0<h<=8 and b*h<=64
        and (t>2048 or (b==1 and t>min_length)) and t<=8192 and d==256
        and k.shape==v.shape and k.shape[0]==b and k.shape[2:]==(t,d) and k.shape[1]>0 and h%k.shape[1]==0
        and b*h*t*d*2<=128*1024*1024 and isinstance(scale,(int,float)) and math.isfinite(scale)
        and all(x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.layout==torch.strided
                and x.stride(-1)==1 and all(s>=0 for s in x.stride()) for x in (q,k,v))
        and mask.device.type=='cpu' and mask.dtype==torch.bool and tuple(mask.shape)==(b,1,1,t)
        and bool(mask.any(dim=-1).all()))

class LongNpuAttention:
    def __init__(self,owner,tile=128,correction=False,min_length=2048):
        if type(min_length) is not int or min_length not in (615,2048):raise ValueError('Unsupported minimum length')
        self.min_length=min_length
        if type(correction) is not bool:raise ValueError('Probability correction must be bool')
        self.correction=correction
        if tile not in (128,512,1024,2048):raise ValueError('Unsupported query tile')
        self.tile=tile
        self.pool=owner.engine.pool
        if self.pool is None:raise ValueError('Owner must provide its joined two-worker pool')
        self.registry=qwen.ALL_ATTENTION_FUNCTIONS;self.previous=self.registry['sdpa']
        self.handles=[];self.contexts=[];self.bucket=None;self.active=False;self.closed=False;self.disabled=False
        self.counts={'query_tile':tile,'min_b1_length_exclusive':min_length,'long_softmax_threads_per_worker':2,'probability_passes':2 if correction else 1,'probability_scale':1 if correction else 1024,'optimized':0,'bypass':0,'fallback':0,'errors':[],'lengths':{},'created':0,'retired_banks':0}
        self.registry.register('sdpa',self.forward)
    def retire(self):
        for handle in self.handles:lib.la_destroy(handle)
        self.handles.clear();self.contexts=[];self.bucket=None
    def ensure(self,length):
        bucket=(length+127)//128*128
        if self.bucket==bucket:return
        if self.handles:self.counts['retired_banks']+=1
        self.retire()
        try:
            for core in (1,2):
                pair=[]
                for k,n in ((256,bucket),(bucket,256)):
                    handle=lib.la_create(self.tile,k,n,core)
                    if not handle:raise RuntimeError('Long attention create: '+lib.la_error().decode())
                    self.handles.append(handle);pair.append(handle);self.counts['created']+=1
                self.contexts.append(pair)
            self.bucket=bucket
        except BaseException:self.retire();raise
    def worker(self,i,q,k,v,mask,out,strides,scale):
        b,h,t,d=q.shape
        rc=lib.la_long_worker(*self.contexts[i],q.data_ptr(),k.data_ptr(),v.data_ptr(),mask.data_ptr(),out.data_ptr(),
            ctypes.addressof(strides),scale,b,h,k.shape[1],t,i,2,int(self.correction))
        if rc==-2:raise ValueError('Long attention nonfinite/range guard')
        if rc:raise RuntimeError('Long attention worker: '+lib.la_error().decode())
    def kernel(self,q,k,v,mask,scale):
        if self.closed or self.active or not supported(q,k,v,mask,scale,self.min_length):raise ValueError('Unsupported long NPU attention signature')
        self.active=True
        try:
            self.ensure(q.shape[2]);b,h,t,d=q.shape
            out=torch.empty((b,t,h,d),dtype=torch.bfloat16);mask=mask.contiguous()
            strides=(ctypes.c_int64*9)(*[s for x in (q,k,v) for s in x.stride()[:3]])
            jobs=[]
            try:
                for i in (0,1):jobs.append(self.pool.submit(self.worker,i,q,k,v,mask,out,strides,scale))
            finally:wait(jobs)
            for job in jobs:job.result()
            if not bool(torch.isfinite(out).all()):raise ValueError('Nonfinite long attention output')
            return out
        finally:self.active=False
    def forward(self,module,query,key,value,attention_mask,*args,**kwargs):
        scale=kwargs.get('scaling',getattr(module,'scaling',None))
        good=(not self.closed and not self.disabled and not self.active and type(module) is qwen.Qwen3_5Attention
            and not module.training and not args and kwargs.get('dropout',0)==0 and not kwargs.get('use_cache',False)
            and not kwargs.get('is_causal',False)
            and all(k in ('dropout','scaling','position_ids','use_cache','is_causal') or v is None for k,v in kwargs.items())
            and supported(query,key,value,attention_mask,scale,self.min_length))
        if good:
            try:result=self.kernel(query,key,value,attention_mask,scale)
            except (ValueError,RuntimeError) as exc:
                self.counts['fallback']+=1;self.counts['errors'].append(str(exc));self.disabled=True
            else:
                self.counts['optimized']+=1;t=str(query.shape[2]);self.counts['lengths'][t]=self.counts['lengths'].get(t,0)+1
                return result,None
        else:self.counts['bypass']+=1
        return self.previous(module,query,key,value,attention_mask,*args,**kwargs)
    def close(self):
        if self.closed:return
        if self.active:raise RuntimeError('Cannot destroy active long attention contexts')
        self.closed=True
        if self.registry['sdpa']==self.forward:self.registry.register('sdpa',self.previous)
        self.retire()
        # The pool belongs to the original Owner and remains available.

def install(owner):
    owner.long_npu_attention=LongNpuAttention(owner)
    return owner.long_npu_attention.close
