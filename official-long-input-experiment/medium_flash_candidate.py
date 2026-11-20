"""Experimental NPU attention retaining the pinned CPU-Flash BF16/PV boundary.

Full original batch, padding and K/V; 512-key online softmax, no model edits.
This candidate is unqualified until the original end-to-end gates pass.
"""
import ctypes,math
from concurrent.futures import wait
import torch
import long_npu_attention_candidate as base

lib=base.lib
lib.la_create_flash.argtypes=[ctypes.c_int]*3;lib.la_create_flash.restype=ctypes.c_void_p
lib.la_flash_worker.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]+[ctypes.c_void_p]*6+[ctypes.c_float]+[ctypes.c_int]*6
lib.la_flash_worker.restype=ctypes.c_int

def supported(q,k,v,mask,scale):
    if any(not isinstance(x,torch.Tensor) for x in (q,k,v,mask)):return False
    if any(x.ndim!=4 for x in (q,k,v)):return False
    b,h,t,d=q.shape
    return (not torch.is_grad_enabled() and 0<b<=8 and 0<h<=8 and b*h<=64 and 615<t<=2048 and d==256
        and k.shape==v.shape and k.shape[0]==b and k.shape[2:]==(t,d) and k.shape[1]>0 and h%k.shape[1]==0
        and isinstance(scale,(int,float)) and math.isfinite(scale)
        and all(x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.layout==torch.strided
                and x.stride(-1)==1 and all(s>=0 for s in x.stride()) for x in (q,k,v))
        and mask.device.type=='cpu' and mask.dtype==torch.bool and tuple(mask.shape)==(b,1,1,t)
        and bool(mask.any(dim=-1).all()))

class MediumFlash(base.LongNpuAttention):
    def __init__(self,owner):
        super().__init__(owner)
        self.counts.update({'domain':'615<T<=2048, original batch <=8','key_block':512,'probability_boundary':'BF16 unnormalised exp before PV'})
    def ensure(self,length):
        blocks=(length+511)//512
        if self.bucket==blocks:return
        if self.handles:self.counts['retired_banks']+=1
        self.retire()
        try:
            for core in (1,2):
                qk=[];pv=[]
                for target,k,n in ((qk,256,512),(pv,512,256)):
                    for _ in range(blocks):
                        handle=lib.la_create_flash(k,n,core)
                        if not handle:raise RuntimeError('Medium flash create: '+lib.la_error().decode())
                        self.handles.append(handle);target.append(handle);self.counts['created']+=1
                self.contexts.append(((ctypes.c_void_p*blocks)(*qk),(ctypes.c_void_p*blocks)(*pv)))
            self.bucket=blocks
        except BaseException:self.retire();raise
    def worker(self,i,q,k,v,mask,out,strides,scale):
        b,h,t,d=q.shape
        rc=lib.la_flash_worker(*self.contexts[i],self.bucket,q.data_ptr(),k.data_ptr(),v.data_ptr(),mask.data_ptr(),out.data_ptr(),ctypes.addressof(strides),scale,b,h,k.shape[1],t,i,2)
        if rc==-2:raise ValueError('Medium flash nonfinite/range guard')
        if rc:raise RuntimeError('Medium flash worker: '+lib.la_error().decode())
    def kernel(self,q,k,v,mask,scale):
        if self.closed or self.active or not supported(q,k,v,mask,scale):raise ValueError('Unsupported medium flash signature')
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
            if not bool(torch.isfinite(out).all()):raise ValueError('Nonfinite medium flash output')
            return out
        finally:self.active=False
    def forward(self,module,query,key,value,attention_mask,*args,**kwargs):
        scale=kwargs.get('scaling',getattr(module,'scaling',None))
        good=(not self.closed and not self.disabled and not self.active and type(module) is base.qwen.Qwen3_5Attention
            and not module.training and not args and kwargs.get('dropout',0)==0 and not kwargs.get('use_cache',False)
            and not kwargs.get('is_causal',False)
            and all(k in ('dropout','scaling','position_ids','use_cache','is_causal') or v is None for k,v in kwargs.items())
            and supported(query,key,value,attention_mask,scale))
        if good:
            try:result=self.kernel(query,key,value,attention_mask,scale)
            except (ValueError,RuntimeError) as exc:
                self.counts['fallback']+=1;self.counts['errors'].append(str(exc));self.disabled=True
            else:
                self.counts['optimized']+=1;t=str(query.shape[2]);self.counts['lengths'][t]=self.counts['lengths'].get(t,0)+1
                return result,None
        else:self.counts['bypass']+=1
        return self.previous(module,query,key,value,attention_mask,*args,**kwargs)

def install(owner):
    owner.medium_flash=MediumFlash(owner)
    return owner.medium_flash.close
