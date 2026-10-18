"""Gate/up use two cores, then independent 320/295-row down projections.

All BF16 intermediate boundaries remain. Two down buffers stay private until
both workers succeed; no context is submitted concurrently with itself.
"""
from native_mlp import NativeMlp
from concurrent.futures import wait
import ctypes,torch
class SplitMlp(NativeMlp):
 def __init__(self,engine,table):
  super().__init__(engine,table,'libnative_split.so');self.split=True
  self.lib.nm_fuse_slice.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p];self.lib.nm_fuse_slice.restype=ctypes.c_int
 def ensure_down(self,module,core):
  key=('split',id(module),core)
  if key not in self.handles:
   w=module.weight.detach().half().t().contiguous()
   if not bool(torch.isfinite(w).all()):raise ValueError('Split weight overflow')
   handle=self.lib.nm_create(module.in_features,module.out_features,w.data_ptr(),core)
   if not handle:raise RuntimeError('Split create '+repr(self.lib.nm_error()))
   self.handles[key]=handle
  return self.handles[key]
 def run(self,members,x):
  if not self.split:return super().run(members,x)
  gate,up,down=members;flat=x.reshape(-1,x.shape[-1]).contiguous();rows=flat.shape[0]
  if rows!=615:return super().run(members,x)
  if gate.out_features!=up.out_features or gate.out_features!=down.in_features or gate.in_features!=down.out_features:raise ValueError('Split MLP shape')
  hg,hu=[self.ensure(m) for m in (gate,up)];hd=[self.ensure_down(down,c) for c in (1,2)]
  self.status(self.lib.nm_prepare(hg,rows,flat.data_ptr()));self.status(self.lib.nm_copy(hu,hg,rows));jobs=[]
  try:
   for h in (hg,hu):jobs.append(self.engine.pool.submit(lambda handle=h:self.status(self.lib.nm_raw(handle,rows))))
  finally:wait(jobs)
  for job in jobs:job.result()
  parts=[(0,320),(320,295)];outputs=[]
  for h,(start,n) in zip(hd,parts):
   self.status(self.lib.nm_fuse_slice(h,hg,hu,start,n,self.table.silu.data_ptr()));outputs.append(torch.empty((n,down.out_features),dtype=torch.bfloat16))
  jobs=[]
  try:
   for h,(_,n),out in zip(hd,parts,outputs):jobs.append(self.engine.pool.submit(lambda handle=h,count=n,value=out:self.status(self.lib.nm_output(handle,count,value.data_ptr()))))
  finally:wait(jobs)
  for job in jobs:job.result()
  self.calls+=1;return torch.cat(outputs,dim=0).reshape(*x.shape[:-1],down.out_features)
