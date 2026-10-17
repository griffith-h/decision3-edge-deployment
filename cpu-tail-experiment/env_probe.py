import argparse,json,time,os
from pathlib import Path
from baseline import load_baseline,stats,state
from lut_ops import Tables
from candidate import Candidate
from validate_candidate import Evaluation,sources
p=argparse.ArgumentParser();p.add_argument('--threads',type=int);p.add_argument('--tag',required=True);p.add_argument('--mkldnn-off',action='store_true');p.add_argument('--count',type=int,default=20);a=p.parse_args()
from choose import cpu_choice
source,selected,cfg=cpu_choice()
if a.threads is not None:cfg['threads']=a.threads
if a.mkldnn_off:cfg['mkldnn']=False
model,engine,counts,close=load_baseline(library='/work/libpipeline_bridge.so' if cfg.get('pipeline') else None);table=Tables();candidate=Candidate(model,engine,table);ev=Evaluation(model);path=Path('/work/logs/environment-'+a.tag+'.json')
r={'config':cfg,'selected_from_screen':selected,'records':[],'sources':sources(),'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OMP_WAIT_POLICY','GOMP_SPINCOUNT','OPENBLAS_NUM_THREADS')}}
try:
 candidate.use(cfg);ev.invoke();ev.invoke();r['settings']=state()
 for i in range(a.count):
  t=time.perf_counter();out=ev.invoke();r['records'].append({'seconds':time.perf_counter()-t,**ev.check(out)})
 r['timing']=stats([v['seconds'] for v in r['records']]);r.update(ev.validate());r['passed'] &= all(v['passed'] for v in r['records']);r['handles']=len(engine.handles)
 path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print('ENV',a.tag,r['timing']['median'],r['passed'],flush=True)
finally:candidate.reset();close()
