"""Close screening-order ambiguity using rotating matched final candidates."""
from application import Application,state,stats
from validate_candidate import sources
from pathlib import Path
import json,time,statistics
root=Path('/work');app=Application();base=json.loads((root/'logs/formal-paired.json').read_text())['selected_config']
configs={'cpu-std':base,'graph-half':dict(base,mlp_graph=True),'graph-half-a72':dict(base,mlp_graph=True,cpu_tune=True)}
r={'settings':state(),'sources':sources(),'configs':configs,'records':[],'scope':'20 rotating rounds; same72 matmul handles;12 persistent graphs; warm115 tokens; threshold .01; no HTTP'}
p=root/'logs/final-graph-comparison.json'
def save():p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
 names=list(configs)
 for i in range(20):
  for n in names[i%3:]+names[:i%3]:
   app.use(configs[n]);app.invoke();before=app.counts();t=time.perf_counter();out=app.invoke();elapsed=time.perf_counter()-t;check=app.check(out);assert check['passed'];counts=app.counts()
   if configs[n].get('mlp_graph'):assert counts['graphs']['optimized']-before['graphs']['optimized']==12 and not counts['graphs']['errors']
   else:assert counts['cpu']['pipeline']['optimized']-before['cpu']['pipeline']['optimized']==12
   r['records'].append({'round':i,'variant':n,'seconds':elapsed,**check,'state':state()});print(i,n,elapsed,flush=True)
  save()
 r['timing']={n:stats([v['seconds'] for v in r['records'] if v['variant']==n]) for n in names};r['paired']={}
 for n in names[1:]:
  pairs=[{v['variant']:v['seconds'] for v in r['records'] if v['round']==i} for i in range(20)]
  r['paired'][n]={'reduction_median':statistics.median(1-x[n]/x['cpu-std'] for x in pairs),'faster_rounds':sum(x[n]<x['cpu-std'] for x in pairs)}
 r['validation']={}
 for n,c in configs.items():
  app.use(c);r['validation'][n]=app.ev.validate();assert r['validation'][n]['passed'];save()
 best=min(names,key=lambda n:r['timing'][n]['median'])
 r['selected']=best if r['timing'][best]['median']<.97*r['timing']['cpu-std']['median'] else 'cpu-std';r['selected_config']=configs[r['selected']];r['passed']=True;save();print('SELECTED',r['selected'],r['paired'],flush=True)
finally:app.close()
