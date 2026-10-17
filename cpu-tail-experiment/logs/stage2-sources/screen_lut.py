from baseline import load_baseline,compare,stats,state,cases
from lut_ops import Tables,install
import json,time,torch,hashlib
from pathlib import Path
model,engine,basecounts,close=load_baseline();table=Tables();restore=None
allcases=cases();single=next(c for c in allcases if c['id']=='single')
refs=json.loads(Path('/attn/logs/final-single-scan.json').read_text())['cases'];fp32=json.loads(Path('/remainder/logs/model-results.json').read_text())['baseline_cases']
extra=json.loads(Path('/attn/logs/final-validate-scan-core.json').read_text())['target_shape_cases']
def invoke(c=single):return model.system_one(**{k:c[k] for k in ('state','questions','images','videos') if k in c})
def check(r,c):
 a=compare(r,refs[c['id']]['response']);b=compare(r,fp32[c['id']]);return {'vs_442':a,'vs_fp32':b,'passed':a['passed'] and b['passed']}
r={'settings':state(),'variants':{},'scope':'screening only; 12 timed fixed115-token requests per variant; same model/72 handles','sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').iterdir() if p.suffix in ('.py','.c','.so')}}
path=Path('/work/logs/lut-screen.json')
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
 invoke();extra_refs={c['id']:invoke(c) for c in extra};r['extra_refs']=extra_refs;r['extra_cases']=extra;save()
 for name in ('baseline','activation','mlp','norm','activation-mlp','activation-mlp-norm'):
  if restore:restore();restore=None
  restore,counters=install(model,engine,table,name);row={'records':[],'cases':{},'extra':{}};r['variants'][name]=row
  invoke();invoke()
  for i in range(12):
   t=time.perf_counter();out=invoke();dt=time.perf_counter()-t;row['records'].append({'seconds':dt,**check(out,single)})
  row['timing']=stats([v['seconds'] for v in row['records']]);row['counts']=dict(counters)
  for c in allcases:
   out=invoke(c);row['cases'][c['id']]={'response':out,**check(out,c)}
  for c in extra:
   out=invoke(c);row['extra'][c['id']]={'response':out,**compare(out,extra_refs[c['id']])}
  row['passed']=all(v['passed'] for v in row['records']+list(row['cases'].values())+list(row['extra'].values()))
  print('SCREEN',name,row['timing']['median'],row['passed'],'delta442',max(v['vs_442']['max_delta'] for v in row['cases'].values()),'extra',max(v['max_delta'] for v in row['extra'].values()),flush=True);save()
 r['handles']=len(engine.handles);r['completed']=True;save()
finally:
 if restore:restore()
 close()
