import sys
sys.path[:0]=['/experiment','/mlp','/work']
from experiment import Owner as OldOwner,compare,state,stats,P
from planned_mlp import PlannedBank
from row_projection import RowProjection
import json,time,copy,torch
class Owner(OldOwner):
 def __init__(self):
  super().__init__();self.projections=None
  try:
   self.bank.close();self.bank=PlannedBank(self.model,root=__import__('pathlib').Path('/mlp'));self.projections=RowProjection(self.model,self.engine)
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('current','mlp_only','single','chunked')
  self.bank.mode='fixed' if mode=='current' else 'planned';self.bank.reset_counts();self.projections.mode=mode if mode in ('single','chunked') else 'off';self.projections.reset_counts()
 def counts(self):return {'mlp':copy.deepcopy(self.bank.counts),'projections':copy.deepcopy(self.projections.counts)}
 def close(self):
  if self.projections:self.projections.close()
  super().close()
