import json,time
from pathlib import Path
if not json.loads(Path('/work/logs/mlp-core-probe.json').read_text())['worth_model_test']:
 print('Combined mask has no measured worthwhile gain; no whole-model promotion',flush=True);raise SystemExit(0)
from baseline import load_baseline,stats,state
from lut_ops import Tables
from candidate import Candidate
from choose import cpu_choice
from validate_candidate import Evaluation,sources
from mlp_graph import HalfGraph,lib
from mlp_subgraph import install
source,selected,cfg=cpu_choice();cfg=dict(cfg,cpu_tune=True)
model,engine,bc,close=load_baseline(library='/work/libpipeline_bridge.so');candidate=Candidate(model,engine,Tables());ev=Evaluation(model);pool={};restore=None
class Borrowed:
 def __init__(self,path):
  key=str(path)
  if key not in pool:
   g=HalfGraph(path);code=lib.mlp_graph_set_core(g.handle,3)
   if code:g.close();raise RuntimeError('combined mask rejected '+str(code))
   pool[key]=g
  self.g=pool[key]
 def run_bf16(self,x):return self.g.run_bf16(x)
 def close(self):pass
r={'settings':state(),'sources':sources(),'config':cfg,'variants':{}};path=Path('/work/logs/mlp-multicore-screen.json')
try:
 for name in ('cpu-a72','mlp-core3'):
  if restore:restore();restore=None
  candidate.use(cfg);counts={}
  if name=='mlp-core3':restore,counts=install(model,Borrowed)
  ev.invoke();ev.invoke();row={'records':[],'config':dict(cfg,mlp_graph=name=='mlp-core3',graph_core=3 if name=='mlp-core3' else None)};r['variants'][name]=row
  for _ in range(16):
   t=time.perf_counter();out=ev.invoke();row['records'].append({'seconds':time.perf_counter()-t,**ev.check(out),'after_state':state()})
  row['timing']=stats([v['seconds'] for v in row['records']]);row['counts']=dict(counts);row.update(ev.validate());row['passed'] &= all(v['passed'] for v in row['records']);print('MLP CORE MODEL',name,row['timing']['median'],row['passed'],flush=True);path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 r['completed']=True;path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
finally:
 if restore:restore()
 candidate.reset()
 for g in pool.values():g.close()
 close()
