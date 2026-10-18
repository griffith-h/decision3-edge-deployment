from selected_owner import *
from pathlib import Path
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());assert refs['completed'];case=refs['cases']['words-512']['request'];current=refs['cases']['words-512']['output'];app=Owner();r={'settings':state(),'execution':app.execution,'records':[],'scope':'Final combined-path controlled ablation; both modes hold the640 bank, only scheduling differs. Original128+128+384 dual handles vs640 graph.'}
try:
 assert app.full_mlp is not None
 app.use('baseline');base=app.invoke(case);r['baseline']=base;r['current']=current
 for enabled in (False,True):app.use('candidate');app.full_mlp.enabled=enabled;app.invoke(case)
 handles=len(app.engine.handles)
 for i in range(12):
  for enabled in ((False,True) if i%2==0 else (True,False)):
   app.use('candidate');app.full_mlp.enabled=enabled;t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t;rec={'full640':enabled,'seconds':dt,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts()};r['records'].append(rec);assert len(app.engine.handles)==handles
   print('MLP ABLATION',enabled,dt,rec['vs_current'],flush=True);(p/'logs/ablate-mlp.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True;(p/'logs/ablate-mlp.json').write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
