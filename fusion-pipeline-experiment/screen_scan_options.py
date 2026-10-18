from scan_owner import *
from pathlib import Path
import os,resource
p=Path('/experiment');grouped=json.loads((p/'execution.json').read_text()).get('grouped',False) if (p/'execution.json').exists() else False;app=Owner(grouped=grouped);case=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['request'];r={'settings':state(),'omp_thread_limit':os.environ.get('OMP_THREAD_LIMIT'),'grouped':grouped,'records':[],'scope':'Native scan options only; Torch/OMP environment3 with thread-limit4 in BOTH modes. Native baseline and wide use explicit3, four/neon explicit4. Same CPU4-7 affinity, model, weights, masks and FP32 cross-block state.'}
try:
 app.use('current');current=app.invoke(case);app.use('baseline');base=app.invoke(case);r['current']=current;r['baseline']=base
 neon_replay=json.loads((p/'logs/replay-neon-scan.json').read_text());neon_passed=neon_replay.get('completed',False) and all(x['allclose'] for x in neon_replay['records'])
 modes=['baseline','wide','four']+(['neon'] if neon_passed else []);r['neon_replay_passed']=neon_passed
 for mode in modes:app.use(mode);app.invoke(case)
 handles=len(app.engine.handles)
 for i in range(8):
  for mode in modes[i%len(modes):]+modes[:i%len(modes)]:
   app.use(mode);t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t;rec={'mode':mode,'seconds':dt,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts()};r['records'].append(rec);assert len(app.engine.handles)==handles
   print('SCAN OPTIONS',mode,dt,rec['vs_baseline'],rec['vs_current'],flush=True);(p/'logs/screen-scan-options.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True;r['handles']=handles;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['final_settings']=state();(p/'logs/screen-scan-options.json').write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
