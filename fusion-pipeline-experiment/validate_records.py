"""Recompute saved output deltas and timing statistics without model inference."""
from pathlib import Path
import json,math,statistics,hashlib
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
 result={'files':{},'passed':False};count=0;failures=[]
 refs=json.loads((P/'logs/references.json').read_text())['cases'];old=json.loads((P/'logs/regression-research-stack.json').read_text())['cases']
 for path in sorted((P/'logs').glob('*.json')):
  j=json.loads(path.read_text());checked=0
  def records(rows,base,current,label):
   nonlocal checked,count
   for rec in rows:
    if 'output' not in rec or 'answers' not in rec['output']:continue
    out=rec['output']
    if 'vs_reference' in rec:check(rec['vs_reference'],compare(out,rows[0]['output']))
    if 'vs_baseline' in rec and base:check(rec['vs_baseline'],compare(out,base))
    if 'vs_current' in rec and current:
     actual=compare(out,current);check(rec['vs_current'],actual)
     if not actual['passed']:failures.append({'file':path.name,'case':label,'mode':rec.get('mode'),'max_delta':actual['max_delta']})
    if 'vs_previous_candidate' in rec and label in old:check(rec['vs_previous_candidate'],compare(out,next(x['output'] for x in old[label]['records'] if x['mode']=='candidate')))
    checked+=1;count+=1
  if 'records' in j and isinstance(j['records'],list):records(j['records'],j.get('baseline',j.get('baseline_reference')),j.get('current',j.get('current_reference')),path.stem)
  if isinstance(j.get('cases'),dict):
   for key,c in j['cases'].items():
    current=c.get('current_reference',c.get('current',refs.get(key,{}).get('output')));base=c.get('baseline_reference',c.get('baseline'))
    if 'records' in c:records(c['records'],base,current,key)
    elif 'output' in c and 'vs_current' in c:records([c],base,current,key)
  if path.name.startswith('formal-') and path.name[7:-5].isdigit():
   assert j['completed'] and j['owner_closed'] and j['engine_handles_after_close']==j['native_handles_after_close']==0
   assert j['fds_before']==j['fds_after'] and j['handles_before']==j['handles_after']
   for mode in ('baseline','candidate'):
    rows=[x for x in j['records'] if x['mode']==mode];assert len(rows)==j['pairs']>=30
    check(j['timing'][mode],stats([x['seconds'] for x in rows]));assert all(x['vs_current']['passed'] for x in rows)
  if path.name=='profile-initial.json':
   for rec in j['model']:assert compare(rec['output'],j['validate_mlp_reference'])['all_numeric_max_delta']==0;checked+=1;count+=1
  if path.name=='profile-selected.json':assert j['completed'] and all(sum(r['mode']==m for r in j['records'])==5 for m in ('baseline','candidate'))
  if checked:result['files'][path.name]=checked
 formal=[json.loads((P/f'logs/formal-{n}.json').read_text()) for n in (1,2,3)]
 assert len({x['run_id'] for x in formal})==len({x['container_hostname'] for x in formal})==3
 result['formal_candidate_timing']=stats([r['seconds'] for f in formal for r in f['records'] if r['mode']=='candidate'])
 result['formal_baseline_timing']=stats([r['seconds'] for f in formal for r in f['records'] if r['mode']=='baseline'])
 final=json.loads((P/'logs/final-regression.json').read_text());assert final['completed'] and final['qualified']
 assert json.loads((P/'service-after.json').read_text())['passed'] and json.loads((P/'logs/artifacts-after.json').read_text())['unchanged']
 seen={};duplicates={};unique_count=0
 for name,n in result['files'].items():
  digest=hashlib.sha256((P/'logs'/name).read_bytes()).hexdigest()
  if digest in seen:duplicates[name]=seen[digest]
  else:seen[digest]=name;unique_count+=n
 result['identical_archive_files']=duplicates;result['record_comparisons_excluding_identical_archives']=unique_count
 result['count_note']='outputs_checked counts saved-record comparisons, including identical archive copies. Neither count is a count of independent inferences; formal inference count is180, including90 candidates.'
 result['outputs_checked']=count;result['preserved_gate_failures']=failures;result['passed']=True;(P/'logs/validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='preserved_gate_failures'},indent=2))
if __name__=='__main__':main()
