# -*- coding: utf-8 -*-
"""LAN HTTP parity, fast/fallback routing, serial ownership and timing validation."""
import argparse,concurrent.futures,datetime,json,math,statistics,time,urllib.request,urllib.error
from pathlib import Path
P=Path(__file__).resolve().parent;repo=P.parent
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8901);p.add_argument('--count',type=int,default=20);p.add_argument('--output',default='staging-http.json');a=p.parse_args()
base=f'http://192.168.1.164:{a.port}';cases=json.loads((P/'http-cases.json').read_text());frozen=json.loads((repo/'cpu-tail-experiment/logs/formal-graph-single.json').read_text());ref442=json.loads((repo/'attention-chunk-experiment/logs/final-single-scan.json').read_text())['cases'];extras=json.loads((repo/'cpu-tail-experiment/logs/lut-screen.json').read_text())['extra_refs']
def call(path,body=None):
 req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 t=time.perf_counter()
 with urllib.request.urlopen(req,timeout=120) as r:d=json.load(r)
 return d,time.perf_counter()-t
def health():return call('/health')[0]
def payload(c):return {k:v for k,v in c.items() if k in ('state','questions','images','videos')}
def numbers(v,path='',confidence=True):
 if isinstance(v,dict):return {a:b for k,x in v.items() if k!='legend' and (confidence or k!='confidence') for a,b in numbers(x,path+'/'+k,confidence).items()}
 if type(v) in (int,float):
  assert math.isfinite(v),(path,v)
  return {path:v}
 return {}
def compare(actual,ref,confidence=True):
 x,y=numbers(actual['answers'],confidence=confidence),numbers(ref['answers'],confidence=confidence);assert x.keys()==y.keys()
 same=all(v.get('choice')==ref['answers'][k].get('choice') for k,v in actual['answers'].items())
 return {'max_delta':max(abs(v-y[k]) for k,v in x.items()),'choices_unchanged':same}
def expected(c):return frozen['cases'].get(c['id'],frozen['extra'].get(c['id']))['response']
def check(c,response):
 r=compare(response,expected(c));assert r['max_delta']<=1e-6 and r['choices_unchanged'],(c['id'],'frozen candidate',r)
 assert response['usage']==expected(c)['usage'] and response['model']==expected(c)['model']
 old=ref442[c['id']]['response'] if c['id'] in ref442 else extras[c['id']]
 original=compare(response,old,False);confidence=compare(response,old,True)
 assert original['max_delta']<=.01 and original['choices_unchanged'],(c['id'],original)
 return {'vs_frozen_including_confidence':r,'vs_442_excluding_confidence':original,'vs_442_including_confidence':confidence}
result={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':base,'cases':{},'records':[],'scope':'warm115-token single-question concurrency1 LAN HTTP, all samples retained; graph fixedB1T115, other shapes fall back'}
def save():(P/a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
for i in range(180):
 try:
  h=health();assert h['status']=='ready' and h['deployment']['release']=='2026-10-18-graph-half-v1';break
 except (OSError,AssertionError):time.sleep(2)
else:raise RuntimeError('Ready timeout')
assert h['deployment']['torch_threads']==3 and h['deployment']['affinity']==[4,5,6,7]
result['initial_health']=h;save()
for c in cases:
 response,seconds=call('/v1/systemone',payload(c));d=health()['deployment'];row={'response':response,'seconds':seconds,**check(c,response),'route':d['last_request']}
 t=response['usage']['input_tokens']
 if t==115:assert row['route']['graph_calls']==12 and row['route']['graph_fallbacks']==0,(c['id'],row['route'])
 else:assert row['route']['graph_calls']==0 and row['route']['graph_fallbacks']>0,(c['id'],row['route'])
 result['cases'][c['id']]=row;save();print('CASE',c['id'],t,row['route'],flush=True)
primary=next(c for c in cases if c['id']=='single');call('/v1/systemone',payload(primary))
for i in range(a.count):
 response,seconds=call('/v1/systemone',payload(primary));check(primary,response);route=health()['deployment']['last_request'];assert route['graph_calls']==12
 result['records'].append({'seconds':seconds,'route':route,**check(primary,response)});print('HTTP',i,seconds,flush=True)
 save()
try:call('/v1/systemone',{'questions':{}});raise AssertionError('Invalid request accepted')
except urllib.error.HTTPError as e:assert e.code==422;result['invalid_request_422']=True
models,_=call('/v1/models');assert models['models'][0]['name']=='vllm-sr/d3-edge';result['models']=models
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 distinct=[cases[-1],cases[-2]];futures=[pool.submit(call,'/v1/systemone',payload(c)) for c in distinct]
 result['concurrent_distinct_requests']=[{'id':c['id'],**check(c,f.result()[0])} for c,f in zip(distinct,futures)]
h=health();assert not h['deployment']['counts']['graphs']['errors'];result['final_health']=h
v=[x['seconds'] for x in result['records']];result['timing']={'count':len(v),'median':statistics.median(v),'p95_nearest_rank':sorted(v)[math.ceil(.95*len(v))-1],'min':min(v),'max':max(v)}
result['passed']=True;save();print('HTTP PASS',a.port,result['timing'],flush=True)
