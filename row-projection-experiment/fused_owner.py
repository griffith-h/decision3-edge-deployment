"""Keep raw pipeline's <=512 cap, extend only the existing safe CPU attention fusion."""
from projected_owner import *
from attention_prepare import install as install_prepare
from types import MethodType
class Owner(Owner):
 def __init__(self):
  self.restore_attention=None;self.restore_conditionals=[];self.restore_mlp=[];self.fused=False;self.guard_mlp=False;self.guard_count=0
  super().__init__()
  original={m:m.forward for m in self.model.backbone.modules() if type(m).__name__=='Qwen3_5Attention'}
  owner=self
  class Facade:
   def eligible(self,m,x):return owner.engine.eligible(m,x) or (owner.fused and id(m) not in owner.projections.disabled and owner.projections.good([m],x))
   def group(self,members,x):
    if x.numel()//x.shape[-1]<=512:return owner.engine.group(members,x)
    return owner.projections.run(members,x,tuple(id(m) for m in members))
  try:
   self.restore_attention,self.attention_counts=install_prepare(self.model,Facade(),self.candidate.table)
   for m,old in original.items():
    prepared=m.forward
    def conditional(layer,*args,_old=old,_prepared=prepared,**kwargs):return _prepared(*args,**kwargs) if self.fused else _old(*args,**kwargs)
    m.forward=MethodType(conditional,m);self.restore_conditionals.append((m,prepared))
   for m in self.model.backbone.modules():
    if type(m).__name__!='Qwen3_5MLP':continue
    old=m.forward
    def mlp_guard(layer,x,_old=old):
     if not self.guard_mlp or x.ndim!=3 or x.shape[0]<=1 or x.shape[0]*x.shape[1]<=512:return _old(x)
     previous=self.bank.mode;self.bank.mode='fixed';self.guard_count+=1
     try:return _old(x)
     finally:self.bank.mode=previous
    m.forward=MethodType(mlp_guard,m);self.restore_mlp.append((m,old))
  except BaseException:self.close();raise
 def use(self,mode):
  self.fused=mode in ('fused','fused_guarded');self.guard_mlp=mode=='fused_guarded';self.guard_count=0
  super().use('chunked' if self.fused or mode=='linear_only' else mode)
  if mode=='linear_only':self.bank.mode='fixed'
  for k in self.attention_counts:self.attention_counts[k]=0
 def counts(self):return {**super().counts(),'attention_fusion':dict(self.attention_counts),'fused_enabled':self.fused,'mlp_policy_fallbacks':self.guard_count}
 def close(self):
  for m,f in reversed(self.restore_mlp):m.forward=f
  self.restore_mlp=[]
  for m,f in self.restore_conditionals:m.forward=f
  self.restore_conditionals=[]
  if self.restore_attention:self.restore_attention();self.restore_attention=None
  super().close()
