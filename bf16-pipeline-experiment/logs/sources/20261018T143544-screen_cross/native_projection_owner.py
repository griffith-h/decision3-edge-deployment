from native_owner import Owner as NativeOwner,state,stats,P,json,time,copy,torch,detailed
class Owner(NativeOwner):
 def __init__(self):
  self.native_projection_enabled=False;self.native_projection_original=None;self.native_projection_disabled=set();self.native_projection_errors=[];self.native_projection_ranges=0;self.native_projection_calls=0
  super().__init__()
  try:
   self.native_projection_original=self.projections.run
   def run(members,x,key):
    good=(self.native_projection_enabled and self.float_active and tuple(x.shape) in ((1,615,1024),(1,615,2048)) and self.projections.good(members,x) and all(m.bias is None and id(m) not in self.native_projection_disabled for m in members))
    if not good:return self.native_projection_original(members,x,key)
    try:out=self.native_mlp.group(members,x)
    except ValueError:
     self.native_projection_ranges+=1;return self.projections.cpu(members,x)
    except RuntimeError as exc:
     self.native_projection_errors.append(str(exc));self.native_projection_disabled.update(key);return self.native_projection_original(members,x,key)
    self.native_projection_calls+=len(members);return out
   self.projections.run=run
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('current','baseline','mlp','candidate')
  super().use('current' if mode=='current' else 'baseline' if mode=='baseline' else 'candidate');self.native_projection_enabled=mode=='candidate';self.native_projection_before=self.native_projection_calls
 def counts(self):return {**super().counts(),'native_projection_calls':self.native_projection_calls-self.native_projection_before,'native_projection_errors':list(self.native_projection_errors),'native_projection_range_fallbacks':self.native_projection_ranges}
 def close(self):
  try:
   if self.native_projection_original:self.projections.run=self.native_projection_original
  finally:super().close()
