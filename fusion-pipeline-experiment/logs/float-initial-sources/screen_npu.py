import sys
sys.path.insert(0,'/tail')
from runner import *
from npu_attention import NpuAttention
import statistics,resource
p=Path('/experiment');app=Owner();native=None;case=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['request'];r={'settings':state(),'records':[],'scope':'Full model pilot; every output computed afresh. Original current and validate_mlp references.'}
try:
 native=NpuAttention();original_att=lk.attention
 app.use('current');current=app.invoke(case);app.use('validate_mlp');baseline=app.invoke(case);r['current']=current;r['baseline']=baseline
 def use(mode):
  app.use('validate_mlp');lk.attention=original_att if mode=='baseline' else native;native.residual=mode.endswith('residual');native.fused=mode.startswith('fused')
 for mode in ('baseline','npu','fused','fused_residual'):use(mode);app.invoke(case)
 for i in range(6):
  modes=['baseline','npu','fused','fused_residual'];modes=modes[i%4:]+modes[:i%4]
  for mode in modes:
   use(mode);t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t;rec={'mode':mode,'seconds':dt,'output':out,'vs_baseline':detailed(out,baseline),'vs_current':detailed(out,current),**app.counts()};r['records'].append(rec);print('NPU SCREEN',mode,dt,rec['vs_baseline'],rec['vs_current'],flush=True);(p/'logs/screen-npu.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['final_settings']=state();(p/'logs/screen-npu.json').write_text(json.dumps(r,indent=2)+'\n')
finally:
 if native:native.close()
 app.close()
