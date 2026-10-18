from native_owner import Owner as NativeOwner,state,stats,P,json,time,copy,torch,detailed
from stream_pipeline import StreamPipeline
class Owner(NativeOwner):
 def __init__(self):
  self.stream=None;self.stream_native_scope=False;self.stream_precise_run=None
  super().__init__()
  try:
   self.stream=StreamPipeline(self);original=self.stream.run
   def scoped(block,x):
    before=self.stream_native_scope;self.stream_native_scope=True
    try:return original(block,x)
    finally:self.stream_native_scope=before
   self.stream.run=scoped;self.stream_precise_run=self.precise_mlp.run
   def run(members,x):
    if self.stream_native_scope and self.native_enabled and self.float_active:return self.native_mlp.run(members,x)
    return self.stream_precise_run(members,x)
   self.precise_mlp.run=run
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('baseline','native','stream')
  super().use('baseline' if mode=='baseline' else 'candidate');self.stream.enabled=mode=='stream';self.stream_before=self.stream.calls;self.chunks_before=self.stream.chunks
 def counts(self):return {**super().counts(),'stream_calls':self.stream.calls-self.stream_before,'stream_chunks':self.stream.chunks-self.chunks_before,'stream_errors':list(self.stream.errors)}
 def close(self):
  try:
   if self.stream:self.stream.close()
   if self.stream_precise_run:self.precise_mlp.run=self.stream_precise_run
  finally:super().close()
