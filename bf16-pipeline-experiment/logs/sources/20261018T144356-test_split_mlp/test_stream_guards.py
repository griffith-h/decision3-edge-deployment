"""Actual-owner scan/consumer faults must join and replay the whole layer."""
from pipeline_owner import *
from pathlib import Path
import threading
p=Path('/experiment');refs=json.loads(Path('/fusion/logs/references.json').read_text());case=refs['cases']['words-512']['request'];app=None;r={'completed':False}
try:
 app=Owner();app.use('baseline');reference=app.invoke(case);stream=app.stream;stream.group=256
 original=stream.lib.d3_stream_scan;fault={'used':False}
 def scan_fault(*args):
  rc=original(*args)
  if not fault['used']:fault['used']=True;return -4
  return rc
 stream.lib.d3_stream_scan=scan_fault;app.use('candidate');out=app.invoke(case);assert out==reference and stream.active is None and len(stream.disabled)==1 and len(stream.errors)==1;r['producer_fault_joined_full_layer_replay']=True
 stream.lib.d3_stream_scan=original;stream.disabled.clear();stream.errors.clear();run=app.precise_mlp.run;fault={'n':0,'used':False}
 def consumer_fault(members,x):
  if threading.current_thread().name=='d3-scan-consumer' and not fault['used']:
   fault['n']+=1
   if fault['n']==2:fault['used']=True;raise RuntimeError('injected second chunk consumer failure')
  return run(members,x)
 app.precise_mlp.run=consumer_fault;app.use('candidate');out=app.invoke(case);assert out==reference and stream.active is None and len(stream.disabled)==1;r['second_chunk_consumer_fault_whole_layer_replay']=True
 before=stream.calls;app.use('candidate');out=app.invoke(case);assert out==reference and stream.calls-before==5 and stream.active is None;r['fault_layer_persistent_bypass']=True
 app.precise_mlp.run=run;stream.errors.clear();app.use('candidate');app.invoke(refs['cases']['single']['request']);assert app.counts()['stream_calls']==0;r['short_shape_bypass']=True
 restores=list(stream.restores);stream.close();stream.close();assert all(layer.forward==old for layer,old in restores);r['stream_forward_restored_double_close']=True;r['completed']=True
finally:
 if app:app.close();app.close();r['engine_handles_after_close']=len(app.engine.handles);r['attention_handles_after_close']=len(app.native_attention.handles);r['joined_after_close']=app.stream.active is None
 (p/'logs/stream-guards.json').write_text(json.dumps(r,indent=2))
print('STREAM GUARDS',r,flush=True)
