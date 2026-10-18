from split_owner import *
from pathlib import Path
import resource
p=Path('/experiment');refs=json.loads(Path('/fusion/logs/references.json').read_text());ref=refs['cases']['words-512'];case=ref['request'];app=None;r={'records':[],'completed':False};path=p/'logs/screen-split.json'
try:
 app=Owner()
 for mode in ('baseline','native','candidate'):
  for _ in range(3):
   app.use(mode);out=app.invoke(case);rec={'mode':mode,'output':out,'vs_current':detailed(out,ref['output']),**app.counts()};r.setdefault('warmups',[]).append(rec);print('NATIVE WARM',mode,rec['native_mlp_calls'],rec['vs_current'],flush=True);assert rec['vs_current']['passed'] and not rec['precise_errors']
 handles=(len(app.engine.handles),len(app.native_mlp.handles))
 for i in range(6):
  for mode in (('baseline','native','candidate') if i%2==0 else ('candidate','native','baseline')):
   app.use(mode);t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t;rec={'pair':i,'mode':mode,'seconds':dt,'output':out,'vs_current':detailed(out,ref['output']),**app.counts()};r['records'].append(rec);path.write_text(json.dumps(r,indent=2));print('NATIVE SCREEN',i,mode,round(dt*1000,2),flush=True);assert rec['vs_current']['passed'] and not rec['precise_errors'] and not rec['precise_range_fallbacks'] and (len(app.engine.handles),len(app.native_mlp.handles))==handles
 r['timing']={m:stats([x['seconds'] for x in r['records'] if x['mode']==m]) for m in ('baseline','native','candidate')};r['completed']=True;r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
finally:
 if app:app.close();r['engine_after_close']=len(app.engine.handles);r['native_after_close']=len(app.native_mlp.handles);r['attention_after_close']=len(app.native_attention.handles)
 path.write_text(json.dumps(r,indent=2))
