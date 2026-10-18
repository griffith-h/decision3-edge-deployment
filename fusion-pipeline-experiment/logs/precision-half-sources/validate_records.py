"""Recompute saved output deltas and timing statistics without model inference."""
from pathlib import Path
import json,math,statistics
P=Path(__file__).resolve().parent

def numeric(x,path='',confidence=False):
 if isinstance(x,dict):return {a:b for k,v in x.items() if k!='legend' and (confidence or k!='confidence') for a,b in numeric(v,path+'/'+k,confidence).items()}
 if type(x) in (int,float):
  assert math.isfinite(x),(path,x)
  return {path:x}
 return {}
def compare(out,ref):
 a,b=numeric(out['answers']),numeric(ref['answers']);assert a.keys()==b.keys();delta={k:abs(v-b[k]) for k,v in a.items()};aa,bb=numeric(out['answers'],confidence=True),numeric(ref['answers'],confidence=True);same=all(v.get('choice')==ref['answers'][k].get('choice') for k,v in out['answers'].items())
 return {'max_delta':max(delta.values()),'confidence_max_delta':max((abs(v-bb[k]) for k,v in aa.items() if k.endswith('/confidence')),default=0),'all_numeric_max_delta':max(abs(v-bb[k]) for k,v in aa.items()),'choices_unchanged':same,'passed':max(delta.values())<=.01 and same,'probability_max_delta':max((v for k,v in delta.items() if '/probabilities/' in k),default=0),'score_max_delta':max((v for k,v in delta.items() if k.endswith('/score')),default=0),'noul_max_delta':max((v for k,v in delta.items() if k.endswith('/noul')),default=0)}
def check(saved,actual):
 for k,v in actual.items():
  assert k in saved,k
  assert saved[k]==v if isinstance(v,bool) else abs(saved[k]-v)<1e-12,(k,saved[k],v)
def stats(xs):
 xs=sorted(xs);return {'count':len(xs),'median':statistics.median(xs),'p95_nearest_rank':xs[math.ceil(len(xs)*.95)-1],'min':xs[0],'max':xs[-1]}
def main():
 result={'files':{},'passed':False};count=0
 for fn in ('screen-wide.json','screen-staging.json','screen-npu.json','screen-combined.json','screen-split.json'):
  path=P/'logs'/fn
  if not path.exists():continue
  j=json.loads(path.read_text());assert j['completed'],fn
  base=j.get('baseline');current=j.get('current');assert base and current
  for rec in j['records']:
   check(rec['vs_baseline'],compare(rec['output'],base));check(rec['vs_current'],compare(rec['output'],current));count+=1
  result['files'][fn]=len(j['records'])
 path=P/'logs/screen-precise.json'
 if path.exists():
  j=json.loads(path.read_text());assert j['completed']
  for case in j['cases'].values():
   for rec in case['records']:check(rec['vs_baseline'],compare(rec['output'],case['baseline']));check(rec['vs_current'],compare(rec['output'],case['current']));count+=1
  result['files'][path.name]=sum(len(x['records']) for x in j['cases'].values())
 # Additional formal/regression files are handled after they are produced.
 result['outputs_checked']=count;result['passed']=True;(P/'logs/validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
