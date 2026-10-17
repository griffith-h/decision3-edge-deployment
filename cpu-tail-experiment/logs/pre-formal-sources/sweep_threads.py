from baseline import load_baseline,stats,state
from lut_ops import Tables
from candidate import Candidate
from validate_candidate import Evaluation,sources
import json,time,statistics
from pathlib import Path
rows=json.loads(Path('/work/logs/pipeline-screen.json').read_text())['variants'];valid={n:r for n,r in rows.items() if r['passed']};selected=min(valid,key=lambda n:valid[n]['timing']['median']);cfg=valid[selected]['config']
model,engine,bc,close=load_baseline(library='/work/libpipeline_bridge.so' if cfg.get('pipeline') else None);candidate=Candidate(model,engine,Tables());ev=Evaluation(model)
configs={f't{t}':dict(cfg,threads=t) for t in (3,1,2,4)};configs['t3-mkldnn-off']=dict(cfg,threads=3,mkldnn=False);r={'configs':configs,'settings':state(),'sources':sources(),'records':[],'variants':{}};path=Path('/work/logs/thread-sweep.json')
try:
 names=list(configs)
 for i in range(12):
  order=names[i%len(names):]+names[:i%len(names)]
  for name in order:
   candidate.use(configs[name]);ev.invoke();started=time.perf_counter();out=ev.invoke();r['records'].append({'round':i,'variant':name,'seconds':time.perf_counter()-started,**ev.check(out),'after_state':state()})
  print('THREAD ROUND',i,flush=True);path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 for name,c in configs.items():
  candidate.use(c);row=ev.validate();row['timing']=stats([v['seconds'] for v in r['records'] if v['variant']==name]);row['config']=c;row['passed'] &= all(v['passed'] for v in r['records'] if v['variant']==name);r['variants'][name]=row;print('THREAD',name,row['timing']['median'],row['passed'],flush=True);path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 r['completed']=True;path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
finally:candidate.reset();close()
