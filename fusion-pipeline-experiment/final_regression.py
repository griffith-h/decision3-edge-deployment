"""Verify the minimal selected Owner and final MLP setting against references.

One fresh candidate per case, plus targeted frozen-baseline equivalence. Saved
old candidate outputs are only comparison references, never returned as answers.
"""
from selected_owner import *
from pathlib import Path
import resource,socket
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());old=json.loads((p/'logs/regression-research-stack.json').read_text());assert refs['completed'] and old['completed'];app=None;finished=False;r={'settings':state(),'container_hostname':socket.gethostname(),'cases':{},'baseline_checks':{},'completed':False,'scope':'Final frozen candidate: one fresh output per case after warmup against original current. Correctness/resource regression only; formal timing in separate containers.'}
try:
 app=Owner();r['execution']=app.execution;r['frozen_plans']=app.frozen_plans
 for key in ('words-512','single','multi-wide','words-921','image'):
  c=refs['cases'][key]['request'];app.use('baseline');out=app.invoke(c);check=detailed(out,old['cases'][key]['baseline_reference']);assert check['all_numeric_max_delta']==0,key;r['baseline_checks'][key]=check
 for key,ref in refs['cases'].items():
  case=ref['request'];app.use('candidate');app.invoke(case);handles=len(app.engine.handles);app.use('candidate');before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(case);seconds=time.perf_counter()-t
  old_candidate=next(x['output'] for x in old['cases'][key]['records'] if x['mode']=='candidate');rec={'family':ref['family'],'request':case,'seconds':seconds,'output':out,'vs_current':detailed(out,ref['output']),'vs_previous_candidate':detailed(out,old_candidate),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before,**app.counts()};assert out['usage']==ref['output']['usage'] and len(app.engine.handles)==handles;r['cases'][key]=rec
  print('SELECTED EQUIVALENCE',key,rec['vs_current'],rec['vs_previous_candidate']['all_numeric_max_delta'],flush=True);(p/'logs/final-regression.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 r['qualified']=all(x['vs_current']['passed'] and not x['full640_errors'] and not x['precise_errors'] and not x['precise_range_fallbacks'] and not x.get('last_readout_errors',[]) and not x['mlp']['errors'] and not x['projections']['errors'] and all(not v['errors'] and not v['native_fallback'] for v in x['long_kernels'].values()) for x in r['cases'].values());assert r['qualified'],'Final candidate failed original-current or runtime gate';r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;finished=True
except BaseException as exc:r['error']=repr(exc);raise
finally:
 try:
  if app:app.close();r['engine_handles_after_close']=len(app.engine.handles);r['native_handles_after_close']=len(app.native_attention.handles);assert r['engine_handles_after_close']==r['native_handles_after_close']==0
  r['owner_closed']=True
 finally:r['completed']=finished and r.get('owner_closed',False);r['final_settings']=state();(p/'logs/final-regression.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
