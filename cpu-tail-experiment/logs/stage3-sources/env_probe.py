import argparse,json,time,os
from pathlib import Path
from baseline import load_baseline,stats,state
from lut_ops import Tables
from candidate import Candidate
from validate_candidate import Evaluation,sources
p=argparse.ArgumentParser();p.add_argument('--threads',type=int,default=3);p.add_argument('--tag',required=True);a=p.parse_args()
configs=json.loads(Path('/work/logs/fusion-screen.json').read_text())['variants'];qualified={k:v for k,v in configs.items() if v['passed']};selected=min(qualified,key=lambda k:qualified[k]['timing']['median']);cfg=dict(qualified[selected]['config'],threads=a.threads)
model,engine,counts,close=load_baseline();table=Tables();candidate=Candidate(model,engine,table);ev=Evaluation(model);path=Path('/work/logs/environment-'+a.tag+'.json')
r={'config':cfg,'selected_from_screen':selected,'records':[],'sources':sources(),'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OMP_WAIT_POLICY','GOMP_SPINCOUNT','OPENBLAS_NUM_THREADS')}}
try:
 candidate.use(cfg);ev.invoke();ev.invoke();r['settings']=state()
 for i in range(20):
  t=time.perf_counter();out=ev.invoke();r['records'].append({'seconds':time.perf_counter()-t,**ev.check(out)})
 r['timing']=stats([v['seconds'] for v in r['records']]);r.update(ev.validate());r['passed'] &= all(v['passed'] for v in r['records']);r['handles']=len(engine.handles)
 path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print('ENV',a.tag,r['timing']['median'],r['passed'],flush=True)
finally:candidate.reset();close()
