from scan_owner import Owner as PreviousOwner,state,stats,P,json,time,copy,torch,detailed
from pathlib import Path
class Owner(PreviousOwner):
 def __init__(self):
  self.execution=json.loads(Path('/experiment/execution.json').read_text());super().__init__(grouped=self.execution['grouped'])
  prior=json.loads(Path('/tail/logs/formal.json').read_text());self.frozen_plans={}
  for case in prior['cases'].values():
   record=case['variants']['baseline']['records'][0];plans=record['mlp'].get('plans',{})
   if len(plans)!=1:continue
   plan=tuple(map(int,next(iter(plans)).split(',')))
   shape=__import__('ast').literal_eval(next(iter(record['mlp']['shapes'])));rows=shape[0]*shape[1]
   self.bank.plans[rows]=plan;self.frozen_plans[str(rows)]=list(plan)
  assert tuple(self.bank.plans[615])==(128,128,384)
 def use(self,mode):
  assert mode in ('current','baseline','candidate')
  super().use(self.execution['mode'] if mode=='candidate' else mode)
  if mode=='candidate' and self.execution.get('residual',False):self.native_attention.residual=True
