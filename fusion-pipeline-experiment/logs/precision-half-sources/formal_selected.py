"""Thirty warm interleaved pairs; repeat in three independent containers."""
from selected_owner import *
from pathlib import Path
import argparse,resource,os,socket,uuid
parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,required=True);parser.add_argument('--pairs',type=int,default=30);a=parser.parse_args();assert a.pairs>=30
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());assert refs['completed'];ref=refs['cases']['words-512'];case=ref['request'];current=ref['output'];app=None;finished=False;r={'run_id':uuid.uuid4().hex,'container_hostname':socket.gethostname(),'round':a.round,'pairs':a.pairs,'settings':state(),'omp_thread_limit':os.environ.get('OMP_THREAD_LIMIT'),'request':case,'current_reference':current,'current_source':'logs/references.json/words-512 fresh original runner/current, equality with prior frozen reference asserted','records':[],'completed':False,'methodology':'Three warmups per mode, alternating pair order; no sample discarded, all outputs recomputed. Independent from instrumented profiles. Both modes hold identical graph/context banks. Warm model method only, no HTTP or concurrency SLA.'};path=p/f'logs/formal-{a.round}.json'
def save():path.write_text(json.dumps(r,indent=2)+'\n')
def fds():return len(list(Path('/proc/self/fd').iterdir()))
try:
 app=Owner();r['execution']=app.execution;r['frozen_baseline_plans']=app.frozen_plans
 for mode in ('baseline','candidate'):
  for _ in range(3):app.use(mode);app.invoke(case)
 app.use('baseline');base=app.invoke(case);r['baseline_reference']=base
 prior=json.loads(Path('/tail/logs/formal.json').read_text())['cases']['words-512']['baseline_reference'];assert base==prior,'Baseline changed from frozen prior output'
 r['handles_before']=len(app.engine.handles);r['fds_before']=fds()
 for i in range(a.pairs):
  for mode in (('baseline','candidate') if i%2==0 else ('candidate','baseline')):
   app.use(mode);before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(case);seconds=time.perf_counter()-t
   assert out['usage']==current['usage'] and out['usage']['input_tokens']==615 and out['model']==current['model']
   rec={'pair':i,'mode':mode,'seconds':seconds,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before,**app.counts()};r['records'].append(rec)
   assert len(app.engine.handles)==r['handles_before'] and fds()==r['fds_before'];assert not rec['full640_errors'] and not rec['precise_errors'] and not rec['mlp']['errors'] and not rec['mlp']['range_fallback'] and not rec['projections']['errors'] and not rec['projections']['range_fallback'] and not rec['projections']['runtime_fallback'] and not rec['engine_cpu_fallbacks'] and all(not v['errors'] and not v['native_fallback'] for v in rec['long_kernels'].values())
   assert rec['vs_baseline']['passed'] and rec['vs_current']['passed'];save()
  print('FORMAL PAIR',a.round,i,[(x['mode'],round(x['seconds']*1000,2)) for x in r['records'][-2:]],flush=True)
 r['timing']={mode:stats([x['seconds'] for x in r['records'] if x['mode']==mode]) for mode in ('baseline','candidate')};r['handles_after']=len(app.engine.handles);r['fds_after']=fds();r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['goal_met_this_round']=r['timing']['candidate']['median']<1 and r['timing']['candidate']['p95_nearest_rank']<1;finished=True
except BaseException as exc:r['error']=repr(exc);raise
finally:
 try:
  if app:
   app.close();r['engine_handles_after_close']=len(app.engine.handles);r['native_handles_after_close']=len(app.original_attention_object.handles);r['grouped_handles_after_close']=len(app.grouped_attention.handles) if app.grouped_attention else 0
   assert r['engine_handles_after_close']==r['native_handles_after_close']==r['grouped_handles_after_close']==0
  r['owner_closed']=True;r['fds_after_close']=fds()
 finally:r['completed']=finished and r.get('owner_closed',False);r['final_settings']=state();save()
