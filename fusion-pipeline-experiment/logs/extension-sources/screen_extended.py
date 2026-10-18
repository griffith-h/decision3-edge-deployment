from extended_owner import *
from pathlib import Path
import resource
p=Path('/experiment');app=Owner();case=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['request'];r={'settings':state(),'records':[],'scope':'Interleaved full-model pilot with same extended native context bank; baseline unchanged512 row tiling, qualified candidate, full615 rows, and full615+column split.'}
try:
 app.use('current');current=app.invoke(case);app.use('baseline');base=app.invoke(case);r['current']=current;r['baseline']=base
 modes=['baseline','attention','qualified','wide','wide_split']
 for mode in modes:app.use(mode);app.invoke(case)
 handles=len(app.engine.handles)
 for i in range(6):
  for mode in modes[i%5:]+modes[:i%5]:
   app.use(mode);t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t;rec={'mode':mode,'seconds':dt,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts()};r['records'].append(rec);assert len(app.engine.handles)==handles
   print('EXTENDED',mode,dt,rec['vs_baseline'],rec['vs_current'],flush=True);(p/'logs/screen-extended.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True;r['handles']=handles;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['final_settings']=state();(p/'logs/screen-extended.json').write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
