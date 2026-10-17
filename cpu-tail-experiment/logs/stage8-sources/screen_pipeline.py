from baseline import load_baseline,stats,state
from lut_ops import Tables
from candidate import Candidate
from validate_candidate import Evaluation,sources
import json,time
from pathlib import Path
prior=json.loads(Path('/work/logs/prepare-screen.json').read_text())['variants'];valid={n:v for n,v in prior.items() if v['passed']};selected=min(valid,key=lambda n:valid[n]['timing']['median']);cfg=valid[selected]['config']
configs={'baseline442':{},'cpu-best':cfg,'pipeline':dict(cfg,pipeline=True),'pipeline-only':{'pipeline':True}}
model,engine,bc,close=load_baseline(library='/work/libpipeline_bridge.so');candidate=Candidate(model,engine,Tables());ev=Evaluation(model);r={'settings':state(),'configs':configs,'variants':{},'sources':sources()};path=Path('/work/logs/pipeline-screen.json')
try:
 for name,cfg in configs.items():
  candidate.use(cfg);ev.invoke();ev.invoke();row={'config':cfg,'records':[]};r['variants'][name]=row
  for i in range(16):
   start=time.perf_counter();out=ev.invoke();row['records'].append({'seconds':time.perf_counter()-start,**ev.check(out),'after_state':state()})
  row['timing']=stats([v['seconds'] for v in row['records']]);row['counts']={k:dict(v) for k,v in candidate.counts.items()};row.update(ev.validate());row['passed'] &= all(v['passed'] for v in row['records'])
  print('PIPELINE',name,row['timing']['median'],row['passed'],'extra',max(v['max_delta'] for v in row['extra'].values()),flush=True);path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 r['completed']=True;path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
finally:candidate.reset();close()
