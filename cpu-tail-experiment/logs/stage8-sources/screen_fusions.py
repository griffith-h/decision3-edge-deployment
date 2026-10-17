from baseline import load_baseline,stats,state
from lut_ops import Tables
from candidate import Candidate
from validate_candidate import Evaluation,sources
import json,time
from pathlib import Path
model,engine,basecounts,close=load_baseline();table=Tables();candidate=Candidate(model,engine,table);ev=Evaluation(model)
previous=json.loads(Path('/work/logs/tail-screen.json').read_text())
choices={n:v for n,v in previous['variants'].items() if v['passed']}
# Best supported CPU attention and small projection are selected only if numerical checks pass.
best_attention=min((n for n in ('attn','attn-scalar') if n in choices),key=lambda n:choices[n]['timing']['median'],default=None)
best_small=min((n for n in ('small','small-merge') if n in choices),key=lambda n:choices[n]['timing']['median'],default=None)
base={'lut':'activation-mlp-norm'}
if best_attention and choices[best_attention]['timing']['median']<.99*choices['baseline']['timing']['median']:base['attention']=0 if best_attention=='attn-scalar' else 1
if best_small and choices[best_small]['timing']['median']<.99*choices['baseline']['timing']['median']:base['small']=best_small=='small-merge'
configs={'baseline':{},'activation-mlp':{'lut':'activation-mlp'},'activation-mlp-norm':{'lut':'activation-mlp-norm'},'base':base,'base-rope':dict(base,rotary=True),'base-residual':dict(base,residual=True),'base-all':dict(base,rotary=True,residual=True)}
r={'settings':state(),'sources':sources(),'variants':{},'configs':configs,'scope':'12-trial screening, not formal result'};path=Path('/work/logs/fusion-screen.json')
try:
 for name,cfg in configs.items():
  candidate.use(cfg);ev.invoke();ev.invoke();row={'config':cfg,'records':[]};r['variants'][name]=row
  for i in range(12):
   t=time.perf_counter();out=ev.invoke();row['records'].append({'seconds':time.perf_counter()-t,**ev.check(out)})
  row['timing']=stats([v['seconds'] for v in row['records']]);row['counts']={k:dict(v) for k,v in candidate.counts.items()};row.update(ev.validate());row['passed'] &= all(v['passed'] for v in row['records'])
  print('FUSION',name,row['timing']['median'],row['passed'],'extra',max(v['max_delta'] for v in row['extra'].values()),flush=True);path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 r['completed']=True;r['handles']=len(engine.handles);path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
finally:candidate.reset();close()
