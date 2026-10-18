from precision_readout_owner import *
from pathlib import Path
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());ref=refs['cases']['words-512'];app=None
r={'records':[],'completed':False,'scope':'Eight rotating interleaved controls after restoring615 BF16 MLP boundaries, identical resident banks.'}
try:
 app=Owner();r['execution']=app.execution;r['current']=ref['output'];app.use('baseline');r['baseline']=app.invoke(ref['request'])
 modes=('precise','precise_readout','precisewide','precisewide_readout')
 for mode in modes:app.use(mode);app.invoke(ref['request'])
 handles=len(app.engine.handles)
 for i in range(8):
  order=modes[i%4:]+modes[:i%4]
  for mode in order:
   app.use(mode);t=time.perf_counter();out=app.invoke(ref['request']);dt=time.perf_counter()-t
   rec={'mode':mode,'seconds':dt,'output':out,'vs_current':detailed(out,r['current']),'vs_baseline':detailed(out,r['baseline']),**app.counts()};r['records'].append(rec);assert len(app.engine.handles)==handles
   print('PRECISION FINAL ABLATION',i,mode,dt,rec['vs_current'],flush=True);(p/'logs/ablate-precision-final.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True
finally:
 if app:app.close()
 (p/'logs/ablate-precision-final.json').write_text(json.dumps(r,indent=2)+'\n')
