from selected_owner import *
from pathlib import Path
import resource
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());assert refs['completed'];app=None;finished=False;r={'settings':state(),'cases':{},'completed':False,'methodology':'Every baseline/candidate output freshly inferred. Current references generated separately with original runner.Owner/current in original3-thread environment. Current gate is authoritative; multi-wide old baseline score failure is preserved.'}
def fds():return len(list(Path('/proc/self/fd').iterdir()))
try:
 app=Owner();r['execution']=app.execution
 for key,ref in refs['cases'].items():
  c=ref['request'];current=ref['output'];row={'family':ref['family'],'request':c,'current_reference':current,'current_source':'logs/references.json/'+key,'records':[]};r['cases'][key]=row
  app.use('baseline');base=app.invoke(c);row['baseline_reference']=base;row['baseline_vs_current']=detailed(base,current)
  for mode in ('baseline','candidate'):app.use(mode);app.invoke(c)
  row['handles_before']=len(app.engine.handles);row['fds_before']=fds()
  n=3 if ref['family'] in ('existing','short_multi_batch') else 1
  for repeat in range(n):
   for mode in (('baseline','candidate') if repeat%2==0 else ('candidate','baseline')):
    app.use(mode);before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(c);elapsed=time.perf_counter()-t
    assert out['usage']==current['usage'] and out['model']==current['model']
    rec={'mode':mode,'seconds':elapsed,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before,**app.counts()};row['records'].append(rec);assert len(app.engine.handles)==row['handles_before'] and fds()==row['fds_before'];(p/'logs/regression.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
  row['handles_after']=len(app.engine.handles);row['fds_after']=fds();cand=[x for x in row['records'] if x['mode']=='candidate'];row['candidate_gate']=all(x['vs_current']['passed'] for x in cand);row['candidate_runtime_gate']=all(not x['full640_errors'] and not x['precise_errors'] and not x['mlp']['errors'] and not x['projections']['errors'] and all(not v['errors'] and not v['native_fallback'] for v in x['long_kernels'].values()) for x in cand)
  print('REGRESSION',key,row['candidate_gate'],row['candidate_runtime_gate'],[(x['mode'],round(x['seconds'],3),x['vs_current']['max_delta']) for x in row['records']],flush=True)
 r['qualified']=all(x['candidate_gate'] and x['candidate_runtime_gate'] for x in r['cases'].values());r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;finished=True
except BaseException as exc:r['error']=repr(exc);raise
finally:
 try:
  if app:app.close()
  r['owner_closed']=True
 finally:
  r['completed']=finished and r.get('owner_closed',False);r['final_settings']=state();(p/'logs/regression.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
