import sys
sys.path.insert(0,'/tail')
from runner import *
from precise_mlp import PreciseMlp
p=Path('/experiment');app=Owner();precise=None;previous=json.loads(Path('/tail/logs/formal.json').read_text());r={'settings':state(),'cases':{},'scope':'Large multi-batch MLP preserves BF16 intermediate boundaries using original native pipeline in <=512 row chunks'}
try:
 precise=PreciseMlp(app.model,app.engine,app.candidate.table)
 for key in ('multi-wide','batch-long'):
  case=previous['cases'][key]['request'];app.use('current');precise.enabled=False;current=app.invoke(case);app.use('validate_mlp');base=app.invoke(case);r['cases'][key]={'request':case,'current':current,'baseline':base,'records':[]}
  for enabled in (False,True):app.use('validate_mlp');precise.enabled=enabled;app.invoke(case)
  for i in range(4):
   for enabled in ((False,True) if i%2==0 else (True,False)):
    app.use('validate_mlp');precise.enabled=enabled;t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t
    rec={'enabled':enabled,'seconds':dt,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),'precise_calls':precise.calls,'precise_chunks':precise.chunks,'precise_errors':list(precise.errors),'precise_range_fallbacks':precise.range_fallbacks,**app.counts()};r['cases'][key]['records'].append(rec);print('PRECISE',key,enabled,dt,rec['vs_current'],flush=True);(p/'logs/screen-precise.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True;(p/'logs/screen-precise.json').write_text(json.dumps(r,indent=2)+'\n')
finally:
 if precise:precise.close()
 app.close()
