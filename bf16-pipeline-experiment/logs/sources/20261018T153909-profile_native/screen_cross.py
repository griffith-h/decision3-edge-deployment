"""Warm interleaved screening against the frozen fifth-stage qualified path."""
from cross_owner import *
from pathlib import Path
import argparse,os
p=Path('/experiment');parser=argparse.ArgumentParser();parser.add_argument('--group',type=int,default=128);parser.add_argument('--threads',type=int,default=3);parser.add_argument('--pairs',type=int,default=6);a=parser.parse_args();refs=json.loads(Path('/fusion/logs/references.json').read_text());ref=refs['cases']['words-512'];case=ref['request'];r={'group':a.group,'threads':a.threads,'settings':state(),'records':[],'completed':False};path=p/f'logs/screen-cross-g{a.group}-t{a.threads}.json';app=None
try:
 app=Owner();app.stream.group=a.group;app.stream.threads=a.threads
 for mode in ('baseline','stream','cross'):
  for _ in range(2):
   app.use(mode);out=app.invoke(case);rec={'mode':mode,'warmup':True,'vs_current':detailed(out,ref['output']),**app.counts()};r.setdefault('warmups',[]).append(rec);path.write_text(json.dumps(r,indent=2));print('WARM',mode,rec['stream_calls'],rec['stream_errors'],rec['vs_current'],flush=True)
   assert rec['vs_current']['passed']
   if mode!='baseline':assert rec['stream_calls']==6 and not rec['stream_errors']
   if mode=='cross':assert rec['cross_hits']==len(app.stream.targets) and rec['cross_pending']==0
 r['actual_transitions']=len(app.stream.targets)
 r['handles_before']=len(app.engine.handles)
 for rep in range(a.pairs):
  for mode in (('baseline','stream','cross') if rep%2==0 else ('cross','stream','baseline')):
   app.use(mode);t=time.perf_counter();out=app.invoke(case);dt=time.perf_counter()-t;rec={'pair':rep,'mode':mode,'seconds':dt,'output':out,'vs_current':detailed(out,ref['output']),**app.counts()};r['records'].append(rec);path.write_text(json.dumps(r,indent=2));print('SCREEN',rep,mode,round(dt*1000,2),rec['stream_calls'],flush=True);assert rec['vs_current']['passed'] and not rec['stream_errors'] and len(app.engine.handles)==r['handles_before']
 r['timing']={m:stats([x['seconds'] for x in r['records'] if x['mode']==m]) for m in ('baseline','stream','cross')};r['completed']=True
finally:
 if app:app.close();r['handles_after_close']=len(app.engine.handles);r['native_after_close']=len(app.native_attention.handles);r['consumer_joined']=app.stream.active is None
 path.write_text(json.dumps(r,indent=2))
