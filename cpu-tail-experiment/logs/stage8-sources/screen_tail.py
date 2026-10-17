from baseline import load_baseline,compare,stats,state,cases
from lut_ops import Tables,install as install_lut
from cpu_attention import install as install_attention
from small_linear import install as install_small
import json,time,torch,hashlib
from pathlib import Path
model,engine,basecounts,close=load_baseline();table=Tables();restores=[]
allcases=cases();single=next(c for c in allcases if c['id']=='single')
refs=json.loads(Path('/attn/logs/final-single-scan.json').read_text())['cases'];fp32=json.loads(Path('/remainder/logs/model-results.json').read_text())['baseline_cases']
prev=json.loads(Path('/work/logs/lut-screen.json').read_text());extra=prev['extra_cases'];extra_refs=prev['extra_refs']
def invoke(c=single):return model.system_one(**{k:c[k] for k in ('state','questions','images','videos') if k in c})
def check(r,c):
 a=compare(r,refs[c['id']]['response']);b=compare(r,fp32[c['id']]);return {'vs_442':a,'vs_fp32':b,'passed':a['passed'] and b['passed']}
r={'settings':state(),'variants':{},'scope':'screening only; same model/72 handles; 12 timed requests per variant','sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').iterdir() if p.suffix in ('.py','.c','.so')}}
qualified={k:v for k,v in prev['variants'].items() if v['passed'] and k!='baseline'}
lut_variant=min(qualified,key=lambda k:qualified[k]['timing']['median']) if qualified else 'mlp'
r['lut_variant']=lut_variant
path=Path('/work/logs/tail-screen.json')
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
def use(name):
 for restore in reversed(restores):restore()
 restores.clear();counters={}
 if 'lut' in name or name in ('activation','mlp-norm'):
  restore,c=install_lut(model,engine,table,lut_variant if 'lut' in name else name);restores.append(restore);counters['lut']=c
 if 'attn' in name:
  restore,c=install_attention(0 if 'scalar' in name else 1);restores.append(restore);counters['attn']=c
 if 'small' in name:
  restore,c=install_small(model,'merge' in name);restores.append(restore);counters['small']=c
 torch.set_num_threads(3)
 return counters
try:
 for name in ('baseline','lut','activation','mlp-norm','attn','attn-scalar','small','small-merge','lut-small-merge','lut-attn','lut-attn-small-merge'):
  counters=use(name);row={'records':[],'cases':{},'extra':{}};r['variants'][name]=row
  invoke();invoke()
  for i in range(12):
   t=time.perf_counter();out=invoke();dt=time.perf_counter()-t;row['records'].append({'seconds':dt,**check(out,single)})
  row['timing']=stats([v['seconds'] for v in row['records']]);row['counts']={k:dict(v) for k,v in counters.items()}
  for c in allcases:
   out=invoke(c);row['cases'][c['id']]={'response':out,**check(out,c)}
  for c in extra:
   out=invoke(c);row['extra'][c['id']]={'response':out,**compare(out,extra_refs[c['id']])}
  row['passed']=all(v['passed'] for v in row['records']+list(row['cases'].values())+list(row['extra'].values()))
  print('SCREEN',name,row['timing']['median'],row['passed'],'delta442',max(v['vs_442']['max_delta'] for v in row['cases'].values()),'extra',max(v['max_delta'] for v in row['extra'].values()),flush=True);save()
 r['handles']=len(engine.handles);r['completed']=True;save()
finally:
 for restore in reversed(restores):restore()
 close()
