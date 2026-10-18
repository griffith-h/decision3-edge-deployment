from fused_owner import *
from pathlib import Path
app=Owner();base=json.loads((P/'logs/screen.json').read_text());r={'settings':state(),'cases':{},'scope':'4 interleaved warmed chunked vs fused samples; current output reference'};path=P/'logs/fused-screen.json'
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
 for key in ['single','words-409','words-410','words-512','multi-wide']:
  c=base['cases'][key]['request'];row={'request':c,'variants':{m:{'records':[]} for m in ['chunked','fused']}};r['cases'][key]=row;ref=base['cases'][key]['reference'];row['reference']=ref;row['input_tokens']=ref['usage']['input_tokens']
  for mode in ['chunked','fused']:app.use(mode);app.invoke(c)
  for repeat in range(4):
   for mode in (['chunked','fused'] if repeat%2==0 else ['fused','chunked']):
    app.use(mode);before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(c);rec={'seconds':time.perf_counter()-t,**compare(out,ref),**app.counts(),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before};row['variants'][mode]['records'].append(rec);save()
  for v in row['variants'].values():v['timing']=stats([x['seconds'] for x in v['records']]);v['passed']=all(x['passed'] for x in v['records'])
  save();print('FUSED SCREEN',key,[(m,v['timing']['median'],v['passed']) for m,v in row['variants'].items()],flush=True)
 r['completed']=True;r['passed']=all(v['variants']['fused']['passed'] for v in r['cases'].values());r['final_settings']=state();save()
finally:app.close()
