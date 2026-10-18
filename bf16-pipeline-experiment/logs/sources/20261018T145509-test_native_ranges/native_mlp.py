"""Own native FP32 context bank, directly fuse native C into native down A."""
import ctypes
from pathlib import Path
from concurrent.futures import wait
import torch
class NativeMlp:
 def __init__(self,engine,table,library="libnative_mlp.so"):
  self.engine=engine;self.table=table;self.handles={};self.closed=False;self.calls=0;self.lib=ctypes.CDLL(str(Path(__file__).with_name(library)))
  self.lib.nm_create.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int];self.lib.nm_create.restype=ctypes.c_void_p;self.lib.nm_destroy.argtypes=[ctypes.c_void_p];self.lib.nm_error.restype=ctypes.c_char_p
  for name,types in [('nm_prepare',[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]),('nm_copy',[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]),('nm_raw',[ctypes.c_void_p,ctypes.c_int]),('nm_fuse',[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p]),('nm_output',[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p])]:getattr(self.lib,name).argtypes=types;getattr(self.lib,name).restype=ctypes.c_int
 def status(self,rc):
  if rc in (-2,-3):raise ValueError('Native MLP finite/range guard '+str(rc))
  if rc:raise RuntimeError('Native MLP '+repr(self.lib.nm_error()))
 def ensure(self,m):
  if self.closed:raise RuntimeError('Native MLP closed')
  key=id(m)
  if key not in self.handles:
   weights=m.weight.detach().half().t().contiguous()
   if not bool(torch.isfinite(weights).all()):raise ValueError('Native weight overflow')
   h=self.lib.nm_create(m.in_features,m.out_features,weights.data_ptr(),self.engine.core.get(key,0))
   if not h:raise RuntimeError('Native create '+repr(self.lib.nm_error()))
   self.handles[key]=h
  return self.handles[key]
 def run(self,members,x):
  gate,up,down=members
  if gate.out_features!=up.out_features or gate.out_features!=down.in_features or gate.in_features!=down.out_features:raise ValueError('Native MLP shape')
  hg,hu,hd=[self.ensure(m) for m in members];flat=x.reshape(-1,x.shape[-1]).contiguous();rows=flat.shape[0]
  self.status(self.lib.nm_prepare(hg,rows,flat.data_ptr()));self.status(self.lib.nm_copy(hu,hg,rows));jobs=[]
  try:
   for h in (hg,hu):jobs.append(self.engine.pool.submit(lambda handle=h:self.status(self.lib.nm_raw(handle,rows))))
  finally:wait(jobs)
  for job in jobs:job.result()
  self.status(self.lib.nm_fuse(hd,hg,hu,rows,self.table.silu.data_ptr()));out=torch.empty((*flat.shape[:-1],down.out_features),dtype=torch.bfloat16);self.status(self.lib.nm_output(hd,rows,out.data_ptr()));self.calls+=1
  return out.reshape(*x.shape[:-1],down.out_features)
 def execute(self,module,x,prepared=False):
  h=self.ensure(module);flat=x.reshape(-1,x.shape[-1]).contiguous();rows=flat.shape[0]
  if not prepared:self.status(self.lib.nm_prepare(h,rows,flat.data_ptr()))
  out=torch.empty((*x.shape[:-1],module.out_features),dtype=torch.bfloat16);self.status(self.lib.nm_output(h,rows,out.data_ptr()));return out
 def group(self,members,x):
  rows=x.numel()//x.shape[-1];flat=x.reshape(-1,x.shape[-1]).contiguous();handles=[self.ensure(m) for m in members]
  self.status(self.lib.nm_prepare(handles[0],rows,flat.data_ptr()))
  for h in handles[1:]:self.status(self.lib.nm_copy(h,handles[0],rows))
  buckets=[[m for m in members if self.engine.bucket.get(id(m),0)==i] for i in range(2)]
  jobs=[]
  try:
   for bucket in buckets:
    if bucket:jobs.append(self.engine.pool.submit(lambda items=bucket:{id(m):self.execute(m,x,True) for m in items}))
  finally:wait(jobs)
  result={}
  for job in jobs:result.update(job.result())
  return result
 def close(self):
  if self.closed:return
  self.closed=True
  for h in self.handles.values():self.lib.nm_destroy(h)
  self.handles.clear()
