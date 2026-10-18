from selected_owner import Owner as SelectedOwner,state,stats,P,json,time,copy,torch,detailed
from stream_pipeline import StreamPipeline
class Owner(SelectedOwner):
 def __init__(self):
  self.stream=None
  super().__init__()
  try:self.stream=StreamPipeline(self)
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('current','baseline','candidate','serial')
  super().use('current' if mode=='current' else 'candidate');self.stream.enabled=mode in ('candidate','serial');self.stream.parallel=mode!='serial';self.stream_before=self.stream.calls;self.chunks_before=self.stream.chunks
 def counts(self):return {**super().counts(),'stream_calls':self.stream.calls-self.stream_before,'stream_chunks':self.stream.chunks-self.chunks_before,'stream_errors':list(self.stream.errors)}
 def close(self):
  try:
   if self.stream:self.stream.close()
  finally:super().close()
