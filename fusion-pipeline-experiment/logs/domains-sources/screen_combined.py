from candidate_owner import *
from pathlib import Path
import statistics,resource
app=Owner();case=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['request'];r={'settings':state(),'records':[],'scope':'Full-model independent interleaved screening. All current and baseline references recomputed.'};p=Path('/experiment');path=p/'logs/screen-combined.json'
try:
 app.use('current');current=app.invoke(case);app.use('baseline');base=app.invoke(case);r['current']=current;r['baseline']=base
 modes=['baseline','attention','mlp640','scan_fused','conv_fused','combined','combined_residual']
 for mode in modes:app.use(mode);app.invoke(case)
 handles=len(app.engine.handles)
 for i in range(6):
  order=modes[i%len(modes):]+modes[:i%len(modes)]
  for mode in order:
   app.use(mode);t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t
   rec={'mode':mode,'seconds':dt,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts()};r['records'].append(rec);assert len(app.engine.handles)==handles
   print('COMBINED',mode,dt,rec['vs_baseline'],rec['vs_current'],flush=True);path.write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True;r['handles']=handles;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['final_settings']=state();path.write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
