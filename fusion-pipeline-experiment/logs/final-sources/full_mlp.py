"""One private 640-row graph per layer, dual-core public graph scheduling.

Only the measured B1/T615 signature is intercepted. On failure, discard the
private result, disable that layer's graph, and replay the original row bank.
"""
from pathlib import Path
import torch
from bucket_graph import HalfGraph,lib
class FullMlp:
 def __init__(self,bank,root=Path('/experiment'),factory=HalfGraph,set_core=None):
  self.bank=bank;self.original=bank.run_rows;self.graphs={};self.disabled=set();self.enabled=False;self.closed=False;self.calls=0;self.errors=[];self.set_core=set_core or (lambda g,mask:lib.row_graph_set_core(g.handle,mask))
  try:
   for i,g in enumerate(bank.graphs):
    h=factory(root/f'mlp-{i}-rows640.rknn');self.graphs[id(g)]=h
    if self.set_core(h,3):raise RuntimeError('640 graph core mask failed')
   bank.run_rows=self.run
  except BaseException:self.close();raise
 def run(self,g,x):
  if not self.enabled or self.bank.mode!='planned' or tuple(x.shape)!=(1,615,1024) or id(g) in self.disabled:return self.original(g,x)
  h=self.graphs[id(g)];padded=torch.zeros((1,640,1024),dtype=x.dtype);padded[:,:615].copy_(x)
  try:out=h.run_bf16(padded)
  except (ValueError,RuntimeError) as exc:
   self.disabled.add(id(g));self.errors.append(str(exc))
   if self.set_core(h,0):raise RuntimeError('Failed resetting faulty 640 graph') from exc
   return self.original(g,x)
  self.calls+=1;self.bank.counts['graph_calls']+=1;self.bank.counts['padded_rows']+=25
  return out[:,:615].contiguous()
 def close(self):
  if self.closed:return
  self.closed=True;self.bank.run_rows=self.original
  first=None
  for h in self.graphs.values():
   try:
    if self.set_core(h,0):raise RuntimeError('640 graph core reset failed')
   except BaseException as exc:
    if first is None:first=exc
   finally:h.close()
  self.graphs={}
  if first:raise first
