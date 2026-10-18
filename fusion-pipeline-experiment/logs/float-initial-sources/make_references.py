"""Fresh original current references using ONLY the unmodified previous Owner.

Run in its own original3-thread container before candidate regressions. Stored
outputs are references for comparison; candidate/baseline always infer afresh.
"""
import sys
sys.path.insert(0,'/tail')
from runner import *
from pathlib import Path
p=Path('/experiment');prior=json.loads(Path('/tail/logs/formal.json').read_text());cases=[]
for key,row in prior['cases'].items():cases.append((key,'existing',row['request']))
extra=json.loads(Path('/row/logs/guarded-regression.json').read_text())['request'];cases.append(('wide2-natural','short_multi_batch',extra))
for i,word in enumerate(('beta','gamma','hello','world','code','math','chat','test','one','two','three','four')):
 c=copy.deepcopy(prior['cases']['words-512']['request']);key='matched615-'+str(i);c.update(id=key,state=c['state'].replace('alpha',word));cases.append((key,'intended_matched615',c))
for c in json.loads(Path('/mlp/http-cases.json').read_text())[7:]:cases.append(('extra-'+c['id'],'short_extra',c))
r={'settings':state(),'source':'Fresh runner.Owner current from /tail, original bridge and original3-thread environment. No candidate Owner or extended bridge instantiated in this process.','cases':{},'completed':False};app=None;finished=False
try:
 app=Owner();app.use('current')
 for key,family,c in cases:
  app.use('current');t=time.perf_counter();out=app.invoke(c);seconds=time.perf_counter()-t
  row={'family':family,'request':c,'output':out,'reference_generation_seconds_not_warm_benchmark':seconds,**app.counts()}
  if key in prior['cases']:
   row['matches_prior_current']=out==prior['cases'][key]['current_reference'];assert row['matches_prior_current'],key
  if family=='intended_matched615':row['actual_tokens']=out['usage']['input_tokens'];row['matched615']=row['actual_tokens']==615
  r['cases'][key]=row;(p/'logs/references.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print('REFERENCE',key,out['usage']['input_tokens'],round(seconds,3),row.get('matched615'),flush=True)
 finished=True
except BaseException as exc:r['error']=repr(exc);raise
finally:
 try:
  if app:app.close()
  r['owner_closed']=True
 finally:
  r['completed']=finished and r.get('owner_closed',False);r['final_settings']=state();(p/'logs/references.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
