from baseline import load_baseline,stats,state
from lut_ops import Tables
from candidate import Candidate
from validate_candidate import Evaluation,sources
from choose import cpu_choice
from mlp_subgraph import install
from mlp_graph import Graph
import torch,json,time
from pathlib import Path
source,selected,cfg=cpu_choice();model,engine,bc,close=load_baseline(library='/work/libpipeline_bridge.so');candidate=Candidate(model,engine,Tables());ev=Evaluation(model);restore=None;pool={}
class Borrowed:
 def __init__(self,path):
  key=str(path)
  if key not in pool:pool[key]=Graph(path)
  self.g=pool[key]
 def run(self,x):return self.g.run(x)
 def close(self):pass
r={'config':cfg,'sources':sources(),'settings':state(),'variants':{},'graph_scope':'all12 MLPs only, fixed B1/T115; original weights; FP16 arithmetic replaces intermediate BF16 roundings; noncausal attention untouched'};path=Path('/work/logs/mlp-graph-screen.json')
try:
 for name in ('baseline442','cpu-best','mlp-graph'):
  if restore:restore();restore=None
  candidate.use({} if name=='baseline442' else cfg);counts={}
  if name=='mlp-graph':restore,counts=install(model,Borrowed)
  ev.invoke();ev.invoke();row={'records':[]};r['variants'][name]=row
  for i in range(16):
   t=time.perf_counter();out=ev.invoke();row['records'].append({'seconds':time.perf_counter()-t,**ev.check(out),'after_state':state()})
  row['timing']=stats([v['seconds'] for v in row['records']]);row['counts']=dict(counts);row.update(ev.validate());row['passed'] &= all(v['passed'] for v in row['records'])
  print('MLP GRAPH',name,row['timing']['median'],row['passed'],'extra',max(v['max_delta'] for v in row['extra'].values()),flush=True);path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 r['completed']=True;path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
finally:
 if restore:restore()
 candidate.reset()
 for g in pool.values():g.close()
 close()
