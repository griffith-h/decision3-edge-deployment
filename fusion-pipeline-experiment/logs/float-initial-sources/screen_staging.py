import sys
sys.path.insert(0,'/tail')
from runner import *
import ctypes,experiment,cpu_backend,statistics,resource
original=experiment.load_baseline
experiment.load_baseline=lambda **kw:original(library='/experiment/libstaging_bridge.so')
backend_get=cpu_backend.Backend.get
def get(self,tuned):
 parts=list(backend_get(self,tuned));parts[-1]=cpu_backend.clone_types(ctypes.CDLL('/experiment/libstaging_bridge.so'),self.initial_engine);return tuple(parts)
cpu_backend.Backend.get=get
p=Path('/experiment');app=Owner();case=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['request'];r={'settings':state(),'records':[],'scope':'Full-model interleaved screening; extra staging copies original raw C bits to reusable private storage before unchanged conversion. Native SDK output unchanged.'}
try:
 app.use('current');current=app.invoke(case);app.use('validate_mlp');base=app.invoke(case)
 r['current']=current;r['baseline']=base
 lib=app.engine.lib;lib.mm_set_staging.argtypes=[ctypes.c_int]
 for width,stage in ((512,0),(640,0),(512,1),(640,1)):app.projections.width=width;lib.mm_set_staging(stage);app.use('validate_mlp');app.invoke(case)
 handles=len(app.engine.handles)
 for i in range(6):
  modes=[(512,0),(640,0),(512,1),(640,1)];modes=modes[i%4:]+modes[:i%4]
  for width,stage in modes:
   app.projections.width=width;lib.mm_set_staging(stage);app.use('validate_mlp');t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t
   rec={'width':width,'stage':stage,'seconds':dt,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts()};r['records'].append(rec)
   assert len(app.engine.handles)==handles;print('WIDE',width,dt,rec['vs_baseline'],rec['vs_current'],flush=True)
   (p/'logs/screen-staging.json').write_text(json.dumps(r,indent=2)+'\n')
 r['handles']=handles;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['completed']=True;r['final_settings']=state();(p/'logs/screen-staging.json').write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
