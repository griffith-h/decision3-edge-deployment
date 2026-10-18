"""Retire unused comparison/half banks at sequential logits shape boundaries.

615 uses the FP32 bank and private native MLP. Other shapes use old half banks.
Do not retain normal FP32 MLP controls alongside both sets of active contexts.
All retirement happens before inference, after the preceding call joined work.
"""
from split_owner import Owner as SplitOwner,state,stats,P,json,time,copy,torch,detailed
from types import MethodType
class Owner(SplitOwner):
 def __init__(self):
  self.cache_closed=False;self.cache_original=None;self.cache_domain=None;self.cache_last_events=[];self.cache_retired={'half':0,'normal_fp32_mlp':0}
  super().__init__()
  try:
   self.cache_original=self.model.logits
   def scope(model,sequences,counts):
    domain='615' if len(sequences)==1 and len(sequences[0])==615 else 'other';self.cache_last_events=[]
    if domain!=self.cache_domain:
     if domain=='615':self.retire('half')
     elif self.native_mlp.handles:self.retire('normal_fp32_mlp')
     self.cache_domain=domain
    return self.cache_original(sequences,counts)
   self.model.logits=MethodType(scope,self.model)
  except BaseException:self.close();raise
 def registry(self):
  modules=[m for m in self.model.backbone.modules() if hasattr(m,'_bp_handle')]
  for packed,_ in self.engine.packed.values():modules.extend(packed)
  return modules
 def retire(self,kind):
  owned=set(self.engine.handles);protected=set(self.float_handles.values());modules=self.registry()
  if kind=='half':retiring={m._bp_handle for m in modules if m._bp_handle is not None and m._bp_handle in owned and m._bp_handle not in protected}
  elif kind=='normal_fp32_mlp':retiring={h for key,h in self.float_handles.items() if key in self.mlp_ids and h in owned}
  else:raise ValueError('Unknown cache bank')
  if not retiring:return
  # Same mixed ABI created every engine-owned handle. Native MLP and attention
  # own separate registries and cannot be freed through this path.
  for m in modules:
   if m._bp_handle in retiring:m._bp_handle=None
  for key,h in list(self.float_handles.items()):
   if h in retiring:del self.float_handles[key]
  self.engine.handles[:]=[h for h in self.engine.handles if h not in retiring]
  for h in retiring:self.mixed.mm_destroy(h)
  self.cache_retired[kind]+=len(retiring);self.cache_last_events.append({'bank':kind,'count':len(retiring)})
 def counts(self):return {**super().counts(),'cache_domain':self.cache_domain,'cache_last_events':list(self.cache_last_events),'cache_retired':dict(self.cache_retired)}
 def close(self):
  if self.cache_closed:return
  self.cache_closed=True
  try:
   if self.cache_original and hasattr(self,'model'):self.model.logits=self.cache_original
  finally:super().close()
