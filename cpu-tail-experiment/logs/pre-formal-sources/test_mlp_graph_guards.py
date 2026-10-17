import torch,json,numpy as np
from pathlib import Path
from types import SimpleNamespace
from mlp_subgraph import install
class Qwen3_5MLP(torch.nn.Module):
 def __init__(self):
  super().__init__();self.gate_proj=torch.nn.Linear(1,1,bias=False);self.up_proj=torch.nn.Linear(1,1,bias=False);self.down_proj=torch.nn.Linear(1,1,bias=False);self.act_fn=torch.nn.SiLU();self.old_calls=0
 def forward(self,x):self.old_calls+=1;return x+1
class Fake:
 owners=[];fail_at=None;mode='ok'
 def __init__(self,path):
  if len(self.owners)==self.fail_at:raise RuntimeError('injected ctor')
  self.closed=0;self.owners.append(self)
 def run(self,x):
  if self.mode=='raise':raise RuntimeError('injected run')
  if self.mode=='nan':return np.full_like(x,np.nan)
  return x.copy()
 def close(self):self.closed+=1
m=torch.nn.Sequential(*(Qwen3_5MLP() for _ in range(12))).bfloat16().eval();model=SimpleNamespace(backbone=m);old=[v.forward for v in m];r={}
restore,c=install(model,Fake)
try:
 with torch.inference_mode():
  x=torch.zeros(1,115,1024,dtype=torch.bfloat16);assert torch.equal(m(x),x);assert c['optimized']==12
  y=torch.zeros(1,116,1024,dtype=torch.bfloat16);assert bool((m(y)==12).all())
  Fake.mode='nan';assert bool((m(x)==12).all());assert c['range_fallback']==12
  Fake.mode='raise';assert bool((m(x)==12).all());assert len(c['errors'])==12
  Fake.mode='ok';assert bool((m(x)==12).all()) # error-disabled contexts remain on old path
 restore();assert all(a.forward==b for a,b in zip(m,old));assert all(v.closed==1 for v in Fake.owners)
 Fake.owners=[];Fake.fail_at=4
 try:install(model,Fake);raise AssertionError('missing ctor error')
 except RuntimeError as e:assert str(e)=='injected ctor'
 assert all(a.forward==b for a,b in zip(m,old));assert all(v.closed==1 for v in Fake.owners)
 r={'passed':True,'counts':c,'partial_construction_cleanup':True}
finally:Fake.mode='ok'
Path('/work/logs/mlp-graph-guards.json').write_text(json.dumps(r,indent=2)+'\n');print('MLP GRAPH GUARDS PASS',flush=True)
