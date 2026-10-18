from readout_owner import *
from pathlib import Path
import resource
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());case=refs['cases']['words-512']['request'];current=refs['cases']['words-512']['output'];app=Owner();r={'settings':state(),'execution':app.execution,'records':[],'scope':'Only final full-attention readout consumer changes. First11 layers process all615 tokens; final K/V length615, query/output row1. Same weights,12 layers, masks and all scan states.'}
try:
 assert app.last_readout.supported
 app.use('baseline');base=app.invoke(case);r['current']=current;r['baseline']=base
 modes=['baseline','candidate','readout']
 for mode in modes:app.use(mode);app.invoke(case)
 handles=len(app.engine.handles)
 for i in range(8):
  for mode in modes[i%3:]+modes[:i%3]:
   app.use(mode);t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t;rec={'mode':mode,'seconds':dt,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts()};r['records'].append(rec);assert len(app.engine.handles)==handles
   print('LAST READOUT',mode,dt,rec['vs_current'],rec['last_readout_calls'],flush=True);(p/'logs/screen-readout.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;(p/'logs/screen-readout.json').write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
