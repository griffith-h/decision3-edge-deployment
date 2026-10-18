from pipeline_owner import Owner as PipelineOwner,state,stats,P,json,time,copy,torch,detailed
from cross_layer import CrossLayer
class Owner(PipelineOwner):
 def __init__(self):
  super().__init__()
  try:self.stream.close();self.stream=CrossLayer(self)
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('baseline','stream','cross')
  super().use('baseline' if mode=='baseline' else 'candidate');self.stream.cross_enabled=mode=='cross';self.stream.pending.clear();self.prefetch_before=self.stream.prefetch_hits
 def counts(self):return {**super().counts(),'cross_hits':self.stream.prefetch_hits-self.prefetch_before,'cross_pending':len(self.stream.pending)}
