from selected_owner import Owner as SelectedOwner,state,stats,P,json,time,copy,torch,detailed
from native_mlp import NativeMlp
class Owner(SelectedOwner):
 def __init__(self):
  self.native_mlp=None;self.native_enabled=False;self.precise_run=None
  super().__init__()
  try:
   self.native_mlp=NativeMlp(self.engine,self.candidate.table);self.precise_run=self.precise_mlp.run
   def run(members,x):
    if self.native_enabled and self.float_active and tuple(x.shape)==(1,615,1024):return self.native_mlp.run(members,x)
    return self.precise_run(members,x)
   self.precise_mlp.run=run
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('current','baseline','candidate')
  super().use('current' if mode=='current' else 'candidate');self.native_enabled=mode=='candidate';self.native_before=self.native_mlp.calls
 def counts(self):return {**super().counts(),'native_mlp_calls':self.native_mlp.calls-self.native_before,'native_mlp_handles':len(self.native_mlp.handles)}
 def close(self):
  try:
   if self.precise_run:self.precise_mlp.run=self.precise_run
   if self.native_mlp:self.native_mlp.close()
  finally:super().close()
