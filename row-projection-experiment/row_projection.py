"""Bounded row tiling around existing public-RKNN projection operations.

No engine eligibility/global parent implementation is widened. Native buffers stay
<=512 rows, complete linear outputs reach the original attention/scan unchanged.
"""
from types import MethodType
import torch

class RowProjection:
 def __init__(self,model,engine,width=512,max_rows=2048):
  if not 1<=width<=512:raise ValueError('Native row cap is 512')
  self.engine=engine;self.width=width;self.max_rows=max_rows;self.mode='off';self.closed=False;self.restores=[];self.disabled=set();self.reset_counts()
  named=dict(model.backbone.named_modules());self.modules={name:m for name,m in named.items() if isinstance(m,torch.nn.Linear) and hasattr(m,'_bp_original') and '.mlp.' not in name}
  self.states={id(m):(m,m.weight,m.weight._version,m.bias,m.bias._version if m.bias is not None else None,m._bp_original) for m in self.modules.values()}
  self.scopes={};self.inventory={'selected':[{'name':n,'k':m.in_features,'n':m.out_features} for n,m in self.modules.items()],'groups':[],'other_linears':[{'name':n,'k':m.in_features,'n':m.out_features,'reason':'MLP graph/original fallback' if '.mlp.' in n else 'not selected by original large-projection adapter'} for n,m in named.items() if isinstance(m,torch.nn.Linear) and n not in self.modules]}
  try:
   for entry in engine.groups:
    if '.mlp' in entry['parent']:continue
    parent=named[entry['parent']];members=[getattr(parent,k) for k in entry['members']]
    if not all(id(m) in self.states for m in members):continue
    scope={'active':False,'used':False,'input':None,'outputs':{},'members':members,'key':tuple(id(m) for m in members)}
    for m in members:self.scopes[id(m)]=scope
    self.inventory['groups'].append(entry)
    old=parent.forward
    def parent_forward(layer,*args,_scope=scope,_old=old,**kwargs):
     if self.mode!='chunked' or _scope['active']:return _old(*args,**kwargs)
     _scope.update(active=True,used=False,input=None,outputs={})
     try:return _old(*args,**kwargs)
     finally:_scope.update(active=False,used=False,input=None,outputs={})
    parent.forward=MethodType(parent_forward,parent);self.restores.append((parent,old))
   for name,m in self.modules.items():
    old=m.forward
    def forward(layer,x,_old=old,_name=name):
     rows=x.numel()//x.shape[-1] if isinstance(x,torch.Tensor) and x.ndim>=2 and x.shape[-1] else 0
     if self.mode=='off' or rows<=512:self.counts['bypass']+=1;return _old(x)
     self.counts['shapes'][str(tuple(x.shape))]=self.counts['shapes'].get(str(tuple(x.shape)),0)+1
     scope=self.scopes.get(id(layer));members=scope['members'] if self.mode=='chunked' and scope is not None and scope['active'] else [layer]
     key=tuple(id(m) for m in members)
     if not self.good(members,x) or any(id(m) in self.disabled for m in members):
      self.counts['guard_fallback']+=1;return _old(x)
     if len(members)>1:
      if not scope['used']:
       scope['used']=True;scope['input']=x;scope['outputs']=self.run(members,x,key)
      if scope['input'] is x and id(layer) in scope['outputs']:return scope['outputs'].pop(id(layer))
     return self.run([layer],x,(id(layer),))[id(layer)]
    m.forward=MethodType(forward,m);self.restores.append((m,old))
  except BaseException:self.close();raise
 def reset_counts(self):self.counts={'bypass':0,'guard_fallback':0,'optimized_projections':0,'wide_groups':0,'chunks':0,'native_max_rows':0,'range_fallback':0,'runtime_fallback':0,'errors':[],'shapes':{},'native_rows':{},'coverage':{}}
 def good(self,members,x):
  if torch.is_grad_enabled() or x.device.type!='cpu' or x.dtype!=torch.bfloat16 or x.layout!=torch.strided or x.ndim not in (2,3):return False
  rows=x.numel()//x.shape[-1]
  if not 512<rows<=self.max_rows:return False
  for m in members:
   original,w,v,b,bv,cpu=self.states[id(m)]
   if m is not original or m.training or m.weight is not w or w._version!=v or w.dtype!=torch.bfloat16 or w.device.type!='cpu' or m.bias is not b or (b is not None and (b._version!=bv or b.dtype!=torch.bfloat16 or b.device.type!='cpu')) or m._forward_hooks or m._forward_pre_hooks or x.shape[-1]!=m.in_features:return False
  return True
 def cpu(self,members,x):return {id(m):self.states[id(m)][-1](x) for m in members}
 def run(self,members,x,key):
  flat=x.reshape(-1,x.shape[-1]);rows=flat.shape[0];outputs={id(m):torch.empty((rows,m.out_features),dtype=x.dtype,device=x.device) for m in members}
  try:
   for start in range(0,rows,self.width):
    part=flat[start:start+self.width];n=part.shape[0];self.counts['chunks']+=1;self.counts['native_max_rows']=max(self.counts['native_max_rows'],n);self.counts['native_rows'][str(n)]=self.counts['native_rows'].get(str(n),0)+1
    before=self.engine.cpu_fallbacks
    value=self.engine.group(members,part) if len(members)>1 else {id(members[0]):self.engine.execute(members[0],part)}
    # The old safe engine can internally CPU-fallback a failed piece. Discard
    # every piece and replay the complete ORIGINAL group, never mixed rounding.
    if self.engine.cpu_fallbacks!=before:raise ValueError('Native per-piece range fallback')
    for m in members:outputs[id(m)][start:start+n].copy_(value[id(m)].reshape(n,m.out_features))
  except ValueError:
   self.counts['range_fallback']+=1;return self.cpu(members,x)
  except RuntimeError as exc:
   self.counts['runtime_fallback']+=1;self.counts['errors'].append(str(exc));self.disabled.update(key);return self.cpu(members,x)
  self.counts['optimized_projections']+=len(members);self.counts['wide_groups']+=int(len(members)>1)
  for m in members:
   name=next(n for n,v in self.modules.items() if v is m);self.counts['coverage'][name]=self.counts['coverage'].get(name,0)+1
  return {id(m):outputs[id(m)].reshape(*x.shape[:-1],m.out_features) for m in members}
 def close(self):
  if self.closed:return
  self.closed=True
  for m,old in reversed(self.restores):m.forward=old
