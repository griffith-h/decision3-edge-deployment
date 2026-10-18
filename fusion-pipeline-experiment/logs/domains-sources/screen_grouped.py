from grouped_owner import *
from pathlib import Path
import resource
p=Path('/experiment');app=Owner();case=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['request'];r={'settings':state(),'records':[],'scope':'Same Owner banks; coalesces two query heads sharing K/V per core, MatMulM1230. Original full K/V and separate per-query normalization.'}
try:
 app.use('current');current=app.invoke(case);app.use('baseline');base=app.invoke(case);r['current']=current;r['baseline']=base
 modes=['baseline','wide','grouped']
 for mode in modes:app.use(mode);app.invoke(case)
 handles=len(app.engine.handles)
 for i in range(8):
  for mode in modes[i%3:]+modes[:i%3]:
   app.use(mode);t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t;rec={'mode':mode,'seconds':dt,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts()};r['records'].append(rec);assert len(app.engine.handles)==handles
   print('GROUPED',mode,dt,rec['vs_baseline'],rec['vs_current'],flush=True);(p/'logs/screen-grouped.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True;r['handles']=handles;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;(p/'logs/screen-grouped.json').write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
