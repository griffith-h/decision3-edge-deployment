import sys
sys.path.insert(0,'/tail')
from runner import *
from split_projection import SplitProjection
from pathlib import Path
p=Path('/experiment');app=Owner();split=None;case=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['request'];r={'settings':state(),'records':[],'scope':'Single output projections split by output columns, original FP16/BF16 boundaries and original512 row tiling. Existing multi-projection groups remain unchanged.'}
try:
 split=SplitProjection(app.projections,app.engine);app.use('current');current=app.invoke(case);app.use('validate_mlp');base=app.invoke(case);r['current']=current;r['baseline']=base
 for enabled in (False,True):split.enabled=enabled;app.use('validate_mlp');app.invoke(case)
 handles=len(app.engine.handles)
 for i in range(8):
  for enabled in ((False,True) if i%2==0 else (True,False)):
   app.use('validate_mlp');split.enabled=enabled;before=split.calls;t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t
   rec={'enabled':enabled,'seconds':dt,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),'split_calls':split.calls-before,**app.counts()};r['records'].append(rec);assert len(app.engine.handles)==handles
   print('SPLIT',enabled,dt,rec['vs_baseline'],rec['vs_current'],flush=True);(p/'logs/screen-split.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True;r['handles']=handles;(p/'logs/screen-split.json').write_text(json.dumps(r,indent=2)+'\n')
finally:
 if split:split.close()
 app.close()
