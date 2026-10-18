import sys
sys.path.insert(0,'/tail')
from runner import *
import ctypes,experiment,cpu_backend,statistics,resource
original=experiment.load_baseline
experiment.load_baseline=lambda **kw:original(library='/experiment/libwide_bridge.so')
backend_get=cpu_backend.Backend.get
def get(self,tuned):
 parts=list(backend_get(self,tuned));parts[-1]=cpu_backend.clone_types(ctypes.CDLL('/experiment/libwide_bridge.so'),self.initial_engine);return tuple(parts)
cpu_backend.Backend.get=get
p=Path('/experiment');app=Owner();case=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['request'];r={'settings':state(),'records':[],'scope':'Same extended dynamic shape bank in both modes; only row scheduling differs. baseline512+103 vs full615->640 padding. Not deployment.'}
try:
 app.use('current');current=app.invoke(case);app.use('validate_mlp');base=app.invoke(case)
 r['current']=current;r['baseline']=base
 for width in (512,640):app.projections.width=width;app.use('validate_mlp');app.invoke(case)
 handles=len(app.engine.handles)
 for i in range(8):
  for width in ((512,640) if i%2==0 else (640,512)):
   app.projections.width=width;app.use('validate_mlp');t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t
   rec={'width':width,'seconds':dt,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts()};r['records'].append(rec)
   assert len(app.engine.handles)==handles;print('WIDE',width,dt,rec['vs_baseline'],rec['vs_current'],flush=True)
   (p/'logs/screen-wide.json').write_text(json.dumps(r,indent=2)+'\n')
 r['handles']=handles;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['completed']=True;r['final_settings']=state();(p/'logs/screen-wide.json').write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
