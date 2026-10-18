from fp32_owner import *
from pathlib import Path
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());ref=refs['cases']['words-512'];app=None;r={'records':[],'completed':False,'scope':'Twelve alternating pairs on qualified FP32615 path. Original CPU BF16-boundary MLP for final1-row readout, no115-row graph padding.'}
try:
 app=Owner();r['execution']=app.execution;r['current']=ref['output'];app.use('baseline');r['baseline']=app.invoke(ref['request'])
 for mode in ('float','float_readout'):app.use(mode);app.invoke(ref['request'])
 handles=len(app.engine.handles)
 for i in range(12):
  for mode in (('float','float_readout') if i%2==0 else ('float_readout','float')):
   app.use(mode);t=time.perf_counter();out=app.invoke(ref['request']);dt=time.perf_counter()-t;rec={'mode':mode,'seconds':dt,'output':out,'vs_current':detailed(out,r['current']),'vs_baseline':detailed(out,r['baseline']),**app.counts()};r['records'].append(rec);assert len(app.engine.handles)==handles and rec['vs_current']['passed']
   print('FLOAT ABLATION',i,mode,dt,rec['vs_current'],flush=True);(p/'logs/ablate-float-final.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True
finally:
 if app:app.close()
 (p/'logs/ablate-float-final.json').write_text(json.dumps(r,indent=2)+'\n')
