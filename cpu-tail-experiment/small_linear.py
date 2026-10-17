"""Cache FP32 copies of unchanged small CPU-only linear weights in inference."""
import torch
from types import MethodType
from lut_ops import good

def install(model,merge=False):
 restores=[];counts={'optimized':0,'fallback':0,'pairs':0}
 scopes={};cache={}
 for parent in model.backbone.modules():
  if not merge or type(parent).__name__!='Qwen3_5GatedDeltaNet':continue
  if not all(hasattr(parent,n) for n in ('in_proj_a','in_proj_b')):continue
  members=[parent.in_proj_a,parent.in_proj_b]
  if not all(isinstance(m,torch.nn.Linear) and not hasattr(m,'_bp_original') and m.bias is None for m in members):continue
  scope={'input':None,'outputs':{},'active':False,'members':members};old=parent.forward
  def forward(self,*args,_old=old,_s=scope,**kwargs):
   if _s['active']:return _old(*args,**kwargs)
   _s.update(input=None,outputs={},active=True)
   try:return _old(*args,**kwargs)
   finally:_s.update(input=None,outputs={},active=False)
  parent.forward=MethodType(forward,parent);restores.append(lambda m=parent,old=old:setattr(m,'forward',old))
  for m in members:scopes[id(m)]=scope
 for m in model.backbone.modules():
  if not isinstance(m,torch.nn.Linear) or hasattr(m,'_bp_original') or m.out_features>64 or m.in_features>2048:continue
  old=m.forward;version=m.weight._version;w=m.weight.detach().float().t().contiguous();b=m.bias.detach().float() if m.bias is not None else None
  cache[id(m)]=(w,b,version,m.bias._version if m.bias is not None else None)
  def linear(self,x,_old=old):
   w,b,version,bversion=cache[id(self)]
   if self.training or not good(x) or self.weight._version!=version or self.weight.device!=x.device or self.weight.dtype!=x.dtype or x.shape[-1]!=self.in_features or (self.bias is not None and (self.bias._version!=bversion or self.bias.device!=x.device or self.bias.dtype!=x.dtype)):
    counts['fallback']+=1;return _old(x)
   scope=scopes.get(id(self))
   if scope is not None and scope['active']:
    if scope['input'] is x and id(self) in scope['outputs']:
     counts['optimized']+=1;return scope['outputs'].pop(id(self))
    members=scope['members']
    if scope['input'] is None and all(cache[id(m)][2]==m.weight._version for m in members):
     key=('pair',id(members[0]));pair=cache.get(key)
     if pair is None:pair=torch.cat([cache[id(m)][0] for m in members],dim=1).contiguous();cache[key]=pair
     outputs=torch.matmul(x.float(),pair).bfloat16().split([m.out_features for m in members],dim=-1)
     scope['input']=x;scope['outputs']={id(m):out for m,out in zip(members,outputs)};counts['pairs']+=1;counts['optimized']+=1
     return scope['outputs'].pop(id(self))
   out=torch.matmul(x.float(),w)
   if b is not None:out=out+b
   counts['optimized']+=1;return out.bfloat16()
  m.forward=MethodType(linear,m);restores.append(lambda m=m,old=old:setattr(m,'forward',old))
 def restore():
  for r in reversed(restores):r()
 return restore,counts
