"""Bounded activation MatMul experiment; CPU FP32 softmax, all keys visible.

Contexts belong to one sequential Owner. Each worker owns disjoint native
buffers. Join every worker before error propagation, reuse, or destruction.
"""
import ctypes
from concurrent.futures import ThreadPoolExecutor,wait
from pathlib import Path
import torch
lib=ctypes.CDLL(str(Path(__file__).with_name('libdynamic_matmul.so')))
lib.dm_error.restype=ctypes.c_char_p
lib.dm_create.argtypes=[ctypes.c_int]*4;lib.dm_create.restype=ctypes.c_void_p
lib.dm_set_b.argtypes=[ctypes.c_void_p]*2;lib.dm_set_b.restype=ctypes.c_int
lib.dm_run.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int];lib.dm_run.restype=ctypes.c_int
lib.dm_destroy.argtypes=[ctypes.c_void_p];lib.dm_destroy.restype=None
lib.dm_attention_worker.argtypes=[ctypes.c_void_p]*8+[ctypes.c_float]+[ctypes.c_int]*4
lib.dm_attention_worker.restype=ctypes.c_int
class NpuAttention:
 def __init__(self,workers=2,residual=False,fused=False):
  if workers not in (1,2):raise ValueError('Only one or two independent cores supported')
  self.workers=workers;self.residual=residual;self.fused=fused;self.handles=[];self.contexts=[];self.pool=None;self.closed=False
  try:
   for i in range(workers):
    pair=[]
    for k,n in ((256,624),(640,256)):
     h=lib.dm_create(615,k,n,i+1)
     if not h:raise RuntimeError(lib.dm_error().decode())
     self.handles.append(h);pair.append(h)
    self.contexts.append(pair)
   self.pool=ThreadPoolExecutor(max_workers=workers,thread_name_prefix='d3-attention')
  except BaseException:self.close();raise
 def check(self,rc):
  if rc:raise RuntimeError(lib.dm_error().decode())
 def __call__(self,q,k,v,mask,scale):
  if self.closed:raise RuntimeError('Closed attention')
  if tuple(q.shape)!=(1,8,615,256) or tuple(k.shape)!=tuple(v.shape) or k.ndim!=4 or k.shape[0]!=1 or k.shape[2:]!=(615,256) or k.shape[1]<=0 or 8%k.shape[1] or any(x.dtype!=torch.bfloat16 or x.device.type!='cpu' for x in (q,k,v)) or any(x.stride(-1)!=1 or any(s<0 for s in x.stride()) for x in (q,k,v)) or mask.device.type!='cpu' or not mask.is_contiguous() or mask.dtype!=torch.bool or tuple(mask.shape)!=(1,1,1,615) or torch.is_grad_enabled():raise ValueError('Unsupported native attention signature')
  if self.fused:
   out=torch.empty((1,615,8,256),dtype=torch.bfloat16);s=(ctypes.c_int64*9)(*[a for x in (q,k,v) for a in x.stride()[:3]])
   def native_worker(i):
    self.check(lib.dm_attention_worker(*self.contexts[i],q.data_ptr(),k.data_ptr(),v.data_ptr(),mask.data_ptr(),out.data_ptr(),ctypes.addressof(s),scale,k.shape[1],i,self.workers,int(self.residual)))
   futures=[]
   try:
    for i in range(self.workers):futures.append(self.pool.submit(native_worker,i))
   finally:wait(futures)
   for future in futures:future.result()
   return out
  # Keep all input protection. Half conversion is exact for finite BF16 values
  # within the supported exponent range; reject instead of clipping overflow.
  q=q.half().contiguous();k=k.half().contiguous();v=v.half().contiguous()
  if not all(bool(torch.isfinite(x).all()) for x in (q,k,v)):raise ValueError('FP16 attention input range')
  if not bool(mask.any()):raise ValueError('All keys masked')
  def worker(i):
   with torch.inference_mode():
    qk,pv=self.contexts[i];values=[];previous=-1;bb=torch.zeros((256,624),dtype=torch.float16);vb=torch.zeros((640,256),dtype=torch.float16)
    for head in range(i,8,self.workers):
     kh=head//(8//k.shape[1])
     if kh!=previous:
      bb[:,:615]=k[0,kh].t();vb[:615]=v[0,kh];self.check(lib.dm_set_b(qk,bb.data_ptr()));self.check(lib.dm_set_b(pv,vb.data_ptr()));previous=kh
     scores=torch.empty((615,624),dtype=torch.float32);self.check(lib.dm_run(qk,q[0,head].data_ptr(),scores.data_ptr(),615))
     scores=scores[:,:615]*scale;scores[:,~mask.reshape(-1)]=-float('inf');prob=torch.softmax(scores,dim=-1)
     aa=torch.zeros((615,640),dtype=torch.float16);aa[:,:615]=prob.half();out=torch.empty((615,256),dtype=torch.float32);self.check(lib.dm_run(pv,aa.data_ptr(),out.data_ptr(),615))
     if self.residual:
      aa[:,:615]=(prob-aa[:,:615].float()).half();low=torch.empty_like(out);self.check(lib.dm_run(pv,aa.data_ptr(),low.data_ptr(),615));out+=low
     if not bool(torch.isfinite(out).all()):raise ValueError('Nonfinite native attention output')
     values.append((head,out.bfloat16()))
    return values
  futures=[]
  try:
   for i in range(self.workers):futures.append(self.pool.submit(worker,i))
  finally:wait(futures)
  values=[x for future in futures for x in future.result()]
  out=torch.empty((1,615,8,256),dtype=torch.bfloat16)
  for head,value in values:out[0,:,head]=value
  return out
 def close(self):
  if self.closed:return
  self.closed=True
  if self.pool:self.pool.shutdown(wait=True)
  for h in reversed(self.handles):lib.dm_destroy(h)
  self.handles=[];self.contexts=[]
