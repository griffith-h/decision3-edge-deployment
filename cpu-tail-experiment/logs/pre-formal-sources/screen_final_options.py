from baseline import load_baseline,stats,state
from candidate import Candidate
from lut_ops import Tables
from validate_candidate import Evaluation,sources
from choose import cpu_choice
from mlp_graph import HalfGraph
from mlp_subgraph import install
import json,time
from pathlib import Path
source,selected,cfg=cpu_choice();model,engine,bc,close=load_baseline(library='/work/libpipeline_bridge.so');candidate=Candidate(model,engine,Tables());ev=Evaluation(model);restore=None;pool={}
class Borrowed:
 def __init__(self,path):
  key=str(path)
  if key not in pool:pool[key]=HalfGraph(path)
  self.g=pool[key]
 def run_bf16(self,x):return self.g.run_bf16(x)
 def close(self):pass
configs={'baseline442':{'cpu_tune':False},'cpu-best':dict(cfg,cpu_tune=False),'cpu-a72':dict(cfg,cpu_tune=True),'mlp-half':dict(cfg,cpu_tune=False,mlp_graph=True),'mlp-half-a72':dict(cfg,cpu_tune=True,mlp_graph=True)}
r={'settings':state(),'sources':sources(),'configs':configs,'variants':{}};path=Path('/work/logs/final-options-screen.json')
try:
 for name,c in configs.items():
  if restore:restore();restore=None
  candidate.use(c);counts={}
  if c.get('mlp_graph'):restore,counts=install(model,Borrowed)
  ev.invoke();ev.invoke();row={'config':c,'records':[]};r['variants'][name]=row
  for i in range(16):
   t=time.perf_counter();out=ev.invoke();row['records'].append({'seconds':time.perf_counter()-t,**ev.check(out),'after_state':state()})
  row['timing']=stats([v['seconds'] for v in row['records']]);row['counts']={'cpu':{k:dict(v) for k,v in candidate.counts.items()},'graphs':dict(counts)};row.update(ev.validate());row['passed'] &= all(v['passed'] for v in row['records'])
  print('FINAL OPTION',name,row['timing']['median'],row['passed'],'extra',max(v['max_delta'] for v in row['extra'].values()),flush=True);path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 r['completed']=True;path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
finally:
 if restore:restore()
 candidate.reset()
 for g in pool.values():g.close()
 close()
