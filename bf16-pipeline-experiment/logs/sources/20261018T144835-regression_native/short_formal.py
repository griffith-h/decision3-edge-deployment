"""Final native MLP candidate: independent warm timing, no instrumentation."""
from accepted_owner import *
from pathlib import Path
import argparse,os,resource,socket,uuid
p=Path('/experiment');parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,required=True);parser.add_argument('--pairs',type=int,default=12);a=parser.parse_args();assert a.pairs>=12;refs=json.loads(Path('/fusion/logs/references.json').read_text());ref=refs['cases']['single'];prior=json.loads(Path('/fusion/logs/final-regression.json').read_text())['cases']['single']['output'];case=ref['request'];app=None;finished=False;r={'run_id':uuid.uuid4().hex,'round':a.round,'pairs':a.pairs,'container_hostname':socket.gethostname(),'settings':state(),'records':[],'completed':False,'methodology':'115-token control: 3 warmups per mode after warming615 banks, alternating pair order, all samples retained; baseline is fifth-stage selected qualified candidate, not validate_mlp. No profiling or hashing during timing.'};path=p/'logs/short-formal.json'
def save():path.write_text(json.dumps(r,indent=2))
def fds():return len(list(Path('/proc/self/fd').iterdir()))
try:
 r['fds_before_owner']=fds();app=Owner()
 for mode in ('baseline','candidate'):
  app.use(mode);app.invoke(refs['cases']['words-512']['request'])
 for mode in ('baseline','candidate'):
  for _ in range(3):app.use(mode);app.invoke(case)
 app.use('baseline');base=app.invoke(case);assert base==prior;r['baseline_reference']=base;r['original_current_reference']=ref['output'];r['handles_before']=[len(app.engine.handles),len(app.native_mlp.handles),len(app.native_attention.handles)];r['fds_before']=fds()
 for rep in range(a.pairs):
  for mode in (('baseline','candidate') if rep%2==0 else ('candidate','baseline')):
   app.use(mode);before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(case);seconds=time.perf_counter()-t;rec={'pair':rep,'mode':mode,'seconds':seconds,'output':out,'vs_current':detailed(out,ref['output']),'vs_baseline':detailed(out,base),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before,**app.counts()};r['records'].append(rec)
   assert rec['vs_current']['passed'] and out['usage']==ref['output']['usage'] and out['usage']['input_tokens']==115
   assert not rec['engine_cpu_fallbacks'] and not rec['precise_errors'] and not rec['precise_range_fallbacks'] and not rec['last_readout_errors'] and not rec['mlp']['errors'] and not rec['projections']['errors'] and not rec['projections']['range_fallback'] and not rec['projections']['runtime_fallback'] and all(not v['errors'] and not v['native_fallback'] for v in rec['long_kernels'].values())
   assert rec['native_mlp_calls']==0 and rec['attention_npu_calls']==0 and rec['last_readout_calls']==0
   assert [len(app.engine.handles),len(app.native_mlp.handles),len(app.native_attention.handles)]==r['handles_before'] and fds()==r['fds_before'];save()
  print('FORMAL',a.round,rep,[(x['mode'],round(x['seconds']*1000,2)) for x in r['records'][-2:]],flush=True)
 r['timing']={m:stats([x['seconds'] for x in r['records'] if x['mode']==m]) for m in ('baseline','candidate')};r['handles_after']=[len(app.engine.handles),len(app.native_mlp.handles),len(app.native_attention.handles)];r['fds_after']=fds();r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;finished=True
except BaseException as exc:r['error']=repr(exc);raise
finally:
 if app:
  app.close();app.close();r['handles_after_close']=[len(app.engine.handles),len(app.native_mlp.handles),len(app.native_attention.handles)];assert r['handles_after_close']==[0,0,0]
 r['final_settings']=state();r['fds_after_close']=fds();r['owner_closed']=True;r['completed']=finished;save()
