import sys
sys.path.insert(0,'/tail')
from runner import *
from full_mlp import FullMlp
from pathlib import Path
r={'passed':False};created=[];fault={'run':False,'core':False};calls=[]
class Graph:
 def __init__(self,path):self.closed=False;self.mask=0;self.calls=0;created.append(self)
 def run_bf16(self,x):
  self.calls+=1
  if fault['run']:raise RuntimeError('injected graph failure')
  return x*3
 def close(self):self.closed=True
class Bank:
 def __init__(self):self.graphs=[object() for _ in range(12)];self.mode='planned';self.counts={'graph_calls':0,'padded_rows':0}
 def run_rows(self,g,x):calls.append(tuple(x.shape));return x*3
def core(g,mask):
 g.mask=mask
 return int(fault['core'] and mask==3 and len(created)==3)
with torch.inference_mode():
 bank=Bank();old=bank.run_rows;owner=FullMlp(bank,factory=Graph,set_core=core);owner.enabled=True;x=torch.randn(1,615,1024).bfloat16();g=bank.graphs[0]
 assert torch.equal(bank.run_rows(g,x),x*3);assert owner.calls==1 and bank.counts=={'graph_calls':1,'padded_rows':25}
 fault['run']=True;assert torch.equal(bank.run_rows(g,x),x*3);assert id(g) in owner.disabled and owner.graphs[id(g)].mask==0;assert len(calls)==1
 assert torch.equal(bank.run_rows(g,x),x*3) and len(calls)==2
 fault['run']=False;small=x[:,:512];assert torch.equal(bank.run_rows(bank.graphs[1],small),small*3)
 owner.close();owner.close();assert bank.run_rows==old and all(g.closed for g in created)
 r['normal_and_fault']={'private_padding':True,'whole_original_bank_fallback':True,'fault_layer_disabled':True,'shape_bypass':True,'restored_original':True,'graphs_closed':len(created)}
 created.clear();fault['core']=True;bank=Bank();old=bank.run_rows
 try:FullMlp(bank,factory=Graph,set_core=core);raise AssertionError('constructor failure ignored')
 except RuntimeError:pass
 assert len(created)==3 and all(g.closed for g in created) and bank.run_rows==old;r['constructor_failure_cleanup']=True
r['passed']=True;Path('/experiment/logs/full-mlp-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('FULL MLP UNIT PASS',r,flush=True)
