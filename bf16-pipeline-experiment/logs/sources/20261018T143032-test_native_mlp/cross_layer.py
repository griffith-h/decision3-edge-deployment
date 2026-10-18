"""Experimental early next-layer QKV submission behind a private full barrier.

Only fixed alternating linear/full layers before the final private readout.
The next full attention still consumes all615 K/V. A prefetch result can only
be consumed by the exact returned hidden tensor, inside its next layer call.
"""
from stream_pipeline import StreamPipeline
from types import MethodType
import torch
class CrossLayer(StreamPipeline):
 def __init__(self,owner):
  self.cross_enabled=False;self.targets={};self.target_states={};self.pending={};self.building=None;self.prefetch_calls=0;self.prefetch_chunks=0;self.prefetch_hits=0;self.cross_restores=[]
  super().__init__(owner)
  try:
   stack=list(owner.model.encoder.stack)
   for i,((previous,full),(layer,nextfull)) in enumerate(zip(stack,stack[1:])):
    if full or not nextfull or i+1==len(stack)-1:continue
    self.targets[id(previous)]=layer;old=layer.forward
    modules=list(layer.modules());self.target_states[id(layer)]=(modules,[(m,m.forward) for m in modules if m is not layer],[(m,n,w,w._version) for m in modules for n,w in m.named_parameters(recurse=False)])
    def forward(block,hidden_states,*args,_old=old,**kwargs):
     cached=self.pending.pop(id(block),None)
     if cached is None or not self.cross_enabled or not self.target_good(block) or hidden_states is not cached['hidden']:return _old(hidden_states,*args,**kwargs)
     norm=block.input_layernorm;normal=norm.forward;projection=self.owner.projections.run;used={'norm':0,'projections':0}
     def normalized(module,x):
      if x is hidden_states:used['norm']+=1;return cached['normalized']
      return normal(x)
     def projected(members,x,key):
      if x is cached['normalized'] and all(id(m) in cached['projections'] for m in members):
       used['projections']+=len(members);return {id(m):cached['projections'][id(m)] for m in members}
      return projection(members,x,key)
     try:
      norm.forward=MethodType(normalized,norm);self.owner.projections.run=projected;out=_old(hidden_states,*args,**kwargs)
     finally:norm.forward=normal;self.owner.projections.run=projection
     if used['norm']!=1 or used['projections']!=3:raise RuntimeError('Prefetch was not consumed exactly once')
     self.prefetch_hits+=1;return out
    layer.forward=MethodType(forward,layer);self.cross_restores.append((layer,old))
   self.after_chunk=self.prefetch
  except BaseException:self.close();raise
 def target_good(self,target):
  modules,methods,weights=self.target_states[id(target)]
  return list(target.modules())==modules and all(not m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks for m in modules) and all(m.forward==fn for m,fn in methods) and all(getattr(m,n) is w and w._version==v and w.device.type=='cpu' and w.dtype==torch.bfloat16 for m,n,w,v in weights)
 def run(self,block,hidden):
  target=self.targets.get(id(block)) if self.cross_enabled else None
  if target is not None:
   modules=list(target.modules())
   if not self.target_good(target):target=None
  if target is None:return super().run(block,hidden)
  members=[target.self_attn.q_proj,target.self_attn.k_proj,target.self_attn.v_proj]
  self.building={'target':target,'members':members,'normalized':torch.empty_like(hidden),'projections':{id(m):torch.empty((1,615,m.out_features),dtype=hidden.dtype) for m in members}}
  try:
   out=super().run(block,hidden);cached=self.building;cached['hidden']=out;self.pending[id(target)]=cached;self.prefetch_calls+=1;return out
  finally:self.building=None
 def prefetch(self,block,part,start,rows):
  cached=self.building
  if cached is None:return
  normalized=cached['target'].input_layernorm(part);outputs=self.owner.engine.group(cached['members'],normalized)
  cached['normalized'][:,start:start+rows].copy_(normalized)
  for m in cached['members']:cached['projections'][id(m)][:,start:start+rows].copy_(outputs[id(m)])
  self.prefetch_chunks+=1
 def close(self):
  try:
   super().close()
  finally:
   for layer,old in reversed(self.cross_restores):layer.forward=old
   self.cross_restores=[];self.pending.clear();self.building=None
