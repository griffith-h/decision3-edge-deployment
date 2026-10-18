"""Preserve BF16 MLP boundaries for qualified multi-batch or explicit615 inputs.

Borrow existing MatMul contexts and worker pool. Private outputs are committed
only after every chunk succeeds; any failure discards the complete MLP result.
"""
from concurrent.futures import wait
from types import MethodType
import torch
from pipeline import configure
class PreciseMlp:
 def __init__(self,model,engine,table,width=512):
  self.long_tokens=0;self.engine=engine;self.table=table;self.width=width;self.enabled=False;self.closed=False;self.restores=[];self.disabled=set();self.calls=0;self.chunks=0;self.errors=[];self.range_fallbacks=0;configure(engine)
  if not 0<width<=640:raise ValueError('Selected bridge cap640')
  try:
   for m in model.backbone.modules():
    if type(m).__name__!='Qwen3_5MLP':continue
    old=m.forward;members=(m.gate_proj,m.up_proj,m.down_proj);weights=tuple((v.weight,v.weight._version,v.bias) for v in members);act=m.act_fn
    def forward(layer,x,_old=old,_members=members,_weights=weights,_act=act):
     good=(self.enabled and not torch.is_grad_enabled() and not layer.training and x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.ndim==3 and x.shape[-1]==1024 and ((x.shape[0]>1 and x.shape[1]<=512 and 512<x.shape[0]*x.shape[1]<=2048) or (self.long_tokens==615 and tuple(x.shape)==(1,615,1024))) and layer.act_fn is _act and type(_act).__name__ in ('SiLUActivation','SiLU') and self.engine.mode=='balanced' and not layer._forward_hooks and not layer._forward_pre_hooks and all(hasattr(v,'_bp_original') and v.weight is w and v.weight._version==ver and v.bias is bias and bias is None and w.dtype==torch.bfloat16 and w.device.type=='cpu' and not v._forward_hooks and not v._forward_pre_hooks for v,(w,ver,bias) in zip(_members,_weights)))
     if not good:return _old(x)
     def cpu():
      gate,up,down=_members
      return down._bp_original(_act(gate._bp_original(x))*up._bp_original(x))
     if id(layer) in self.disabled:return cpu()
     try:out=self.run(_members,x)
     except ValueError:self.range_fallbacks+=1;return cpu()
     except RuntimeError as exc:self.errors.append(str(exc));self.disabled.add(id(layer));return cpu()
     self.calls+=1;return out
    m.forward=MethodType(forward,m);self.restores.append((m,old))
  except BaseException:self.close();raise
 def run(self,members,x):
  gate,up,down=members
  if gate.out_features!=up.out_features or gate.out_features!=down.in_features or gate.in_features!=1024 or down.out_features!=1024:raise ValueError('Unexpected MLP dimensions')
  # Equal gate/up projections use the original independent handles.
  hg,hu,hd=(self.engine.ensure(m) for m in members);flat=x.reshape(-1,1024);out=torch.empty_like(flat)
  for start in range(0,flat.shape[0],self.width):
   part=flat[start:start+self.width].contiguous();rows=part.shape[0]
   self.engine.status(self.engine.lib.mm_prepare_bf16(hg,rows,part.data_ptr()));self.engine.status(self.engine.lib.mm_copy_prepared(hu,hg,rows))
   def raw(handle):self.engine.status(self.engine.lib.mm_run_raw(handle,rows))
   jobs=[]
   try:
    for h in (hg,hu):jobs.append(self.engine.pool.submit(raw,h))
   finally:wait(jobs)
   for job in jobs:job.result()
   rc=self.engine.lib.mm_prepare_mlp(hd,hg,hu,rows,self.table.silu.data_ptr())
   if rc in (-2,-3):raise ValueError('MLP intermediate range')
   self.engine.status(rc);value=torch.empty_like(part);rc=self.engine.lib.mm_run_prepared_bf16(hd,rows,None,value.data_ptr())
   if rc in (-2,-3):raise ValueError('MLP output range')
   self.engine.status(rc);out[start:start+rows].copy_(value);self.engine.calls+=3;self.chunks+=1
  return out.reshape(x.shape)
 def close(self):
  if self.closed:return
  self.closed=True
  for m,old in reversed(self.restores):m.forward=old
  self.restores=[]
