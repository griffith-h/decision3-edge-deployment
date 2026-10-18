from native_owner import Owner as NativeOwner,state,stats,P,json,time,copy,torch,detailed
from split_mlp import SplitMlp
class Owner(NativeOwner):
 def __init__(self):
  super().__init__()
  try:self.native_mlp.close();self.native_mlp=SplitMlp(self.engine,self.candidate.table)
  except BaseException:self.close();raise
 def use(self,mode):
  assert mode in ('current','baseline','native','candidate')
  super().use('current' if mode=='current' else 'baseline' if mode=='baseline' else 'candidate');self.native_mlp.split=mode=='candidate'
