from selected_owner import Owner as PreviousOwner,state,stats,P,json,time,copy,torch,detailed
from last_readout import LastReadout
class Owner(PreviousOwner):
 def __init__(self):
  self.last_readout=None
  super().__init__()
  try:self.last_readout=LastReadout(self)
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('current','baseline','candidate','readout')
  super().use('candidate' if mode=='readout' else mode);self.last_readout.enabled=mode=='readout';self.readout_before=self.last_readout.calls;self.bypass_before=self.last_readout.bypasses
 def counts(self):return {**super().counts(),'last_readout_calls':self.last_readout.calls-self.readout_before,'last_readout_bypasses':self.last_readout.bypasses-self.bypass_before,'last_readout_errors':list(self.last_readout.errors),'last_readout_metadata':dict(self.last_readout.last_metadata)}
 def close(self):
  try:
   if self.last_readout:self.last_readout.close()
  finally:super().close()
