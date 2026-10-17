from baseline import load_baseline,stats,state
from lut_ops import Tables
from candidate import Candidate
from validate_candidate import Evaluation,sources
from gpu_client import GPU
from gpu_ops import install_attention,install_scan
import json,time
from pathlib import Path
prior=json.loads(Path('/work/logs/fusion-screen.json').read_text())['variants'];valid={n:v for n,v in prior.items() if v['passed']};selected=min(valid,key=lambda n:valid[n]['timing']['median']);cfg=valid[selected]['config']
model,engine,basecounts,close=load_baseline();candidate=Candidate(model,engine,Tables());ev=Evaluation(model);gpu=GPU();restores=[]
r={'settings':state(),'config':cfg,'cpu_selected':selected,'variants':{},'sources':sources(),'scope':'GPU worker CPU0..3 in private host namespace, inference CPU4..7, Unix/shared-memory copies included; 12-trial screening'};path=Path('/work/logs/gpu-screen.json')
unit=json.loads(Path('/work/logs/gpu-unit.json').read_text())
scan_replay=sum(v['timing']['median'] for v in unit['replay'])
variants=['baseline442','cpu-best','gpu-attention']
if scan_replay<.06:variants.extend(['gpu-scan','gpu-both'])
else:r['skipped_for_performance']={'gpu-scan':scan_replay,'gpu-both':scan_replay,'reason':'sum of six GPU scan replay medians alone already exceeds the entire model budget; includes IPC/copies, no end-to-end GPU-scan probability claim'}
try:
 for name in variants:
  for restore in reversed(restores):restore()
  restores.clear();candidate.use({} if name=='baseline442' else cfg);counts={}
  if name in ('gpu-attention','gpu-both'):restore,c=install_attention(gpu);restores.append(restore);counts['attention']=c
  if name in ('gpu-scan','gpu-both'):restore,c=install_scan(gpu);restores.append(restore);counts['scan']=c
  ev.invoke();ev.invoke();row={'records':[]};r['variants'][name]=row
  for i in range(12):
   t=time.perf_counter();out=ev.invoke();row['records'].append({'seconds':time.perf_counter()-t,**ev.check(out)})
  row['timing']=stats([v['seconds'] for v in row['records']]);row['counts']={k:dict(v) for k,v in counts.items()};row.update(ev.validate());row['passed'] &= all(v['passed'] for v in row['records'])
  print('GPU MODEL',name,row['timing']['median'],row['passed'],'extra',max(v['max_delta'] for v in row['extra'].values()),flush=True);path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 r['completed']=True;path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
finally:
 for restore in reversed(restores):restore()
 candidate.reset();gpu.close();close()
