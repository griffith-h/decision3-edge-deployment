"""Use an independent extended-shape bridge; preserve all old global eligibility.

Only the exact measured B1/T615 row-projection signatures select a640 row
submission. Other lengths retain512-row chunks and their prior scheduling.
"""
from candidate_owner import Owner as PreviousOwner,state,stats,P,json,time,copy,torch,detailed
from split_projection import SplitProjection
import experiment,cpu_backend,ctypes
class Owner(PreviousOwner):
 def __init__(self):
  self.split=None;self.row_original=None;self.wide_enabled=False
  load=experiment.load_baseline;get=cpu_backend.Backend.get
  experiment.load_baseline=lambda **kw:load(library='/experiment/libwide_bridge.so')
  def wide_get(backend,tuned):
   parts=list(get(backend,tuned));parts[-1]=cpu_backend.clone_types(ctypes.CDLL('/experiment/libwide_bridge.so'),backend.initial_engine);return tuple(parts)
  cpu_backend.Backend.get=wide_get
  try:super().__init__()
  finally:experiment.load_baseline=load;cpu_backend.Backend.get=get
  try:
   self.split=SplitProjection(self.projections,self.engine,max_rows=640,min_rows=513)
   self.row_original=self.projections.run
   def run(members,x,key):
    previous=self.projections.width
    self.projections.width=640 if self.wide_enabled and tuple(x.shape) in ((1,615,1024),(1,615,2048)) else 512
    try:return self.row_original(members,x,key)
    finally:self.projections.width=previous
   self.projections.run=run
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('current','baseline','attention','qualified','wide','wide_split')
  super().use(mode if mode in ('current','baseline','attention') else 'qualified')
  self.wide_enabled=mode in ('wide','wide_split');self.split.enabled=mode=='wide_split';self.split_before=self.split.calls
 def counts(self):return {**super().counts(),'wide_enabled':self.wide_enabled,'split_calls':self.split.calls-self.split_before}
 def close(self):
  try:
   if self.row_original and hasattr(self,'projections') and self.projections:self.projections.run=self.row_original
   if self.split:self.split.close()
  finally:super().close()
