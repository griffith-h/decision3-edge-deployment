"""Real native buffers: range, joined faults, persistent CPU, and cleanup."""
import sys
sys.path.insert(0,'/tail')
from runner import *
from split_mlp import SplitMlp
from precise_mlp import PreciseMlp
from lut_ops import Tables
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pathlib import Path
import ctypes,threading
p=Path('/experiment');torch.set_num_threads(3);torch.manual_seed(12);events=[]
def init(self):
 torch.nn.Module.__init__(self);self.gate_proj=torch.nn.Linear(1024,128,bias=False,dtype=torch.bfloat16);self.up_proj=torch.nn.Linear(1024,128,bias=False,dtype=torch.bfloat16);self.down_proj=torch.nn.Linear(128,1024,bias=False,dtype=torch.bfloat16);self.act_fn=torch.nn.SiLU()
 for m in (self.gate_proj,self.up_proj,self.down_proj):m._bp_original=m.forward
 def forbidden(x):raise AssertionError('Old graph fallback must not run')
 # Precise wrapper owns original CPU linears independently of this old forward.
 self.old_graph=forbidden
def forward(self,x):return self.old_graph(x)
MLP=type('Qwen3_5MLP',(torch.nn.Module,),{'__init__':init,'forward':forward});model=MLP().eval();lib=ctypes.CDLL('/fusion/libmixed_bridge.so');engine=SimpleNamespace(mode='balanced',lib=lib,pool=ThreadPoolExecutor(2),core={},bucket={},calls=0);table=Tables();native=SplitMlp(engine,table);precise=PreciseMlp(SimpleNamespace(backbone=model),engine,table,width=640);precise.long_tokens=615;precise.enabled=True;precise.run=lambda members,x:native.run(members,x);r={'completed':False};done=threading.Event();original_raw=native.lib.nm_raw;counter={'raw':0}
def cpu(x):return model.down_proj._bp_original(model.act_fn(model.gate_proj._bp_original(x))*model.up_proj._bp_original(x))
try:
 with torch.inference_mode():
  x=torch.randn(1,615,1024).bfloat16();expected=cpu(x);members=(model.gate_proj,model.up_proj,model.down_proj);hg,hu,hd=[native.ensure(m) for m in members]
  def fault(h,rows):
   counter['raw']+=1
   if h==hg:return -1
   time.sleep(.02);result=original_raw(h,rows);done.set();return result
  native.lib.nm_raw=fault;out=model(x);assert done.is_set() and torch.equal(out,expected) and id(model) in precise.disabled;r['runtime_joined_whole_cpu_fallback']=True
  before=counter['raw'];assert torch.equal(model(x),expected) and counter['raw']==before;r['runtime_disabled_persistently_cpu']=True
  native.lib.nm_raw=original_raw;precise.disabled.clear();huge=torch.full_like(x,100000.);expected=cpu(huge);assert torch.isfinite(expected).all();assert torch.equal(model(huge),expected) and precise.range_fallbacks==1;r['input_overflow_whole_cpu_fallback']=True
  # Subsequent valid inputs still use the range-rejected, unpoisoned contexts.
  before=native.calls;out=model(x);assert native.calls==before+1 and torch.isfinite(out).all();r['range_guard_recoverable']=True
  # Normal/group native output is private; wait every peer on error.
  engine.bucket={id(model.gate_proj):0,id(model.up_proj):1};done.clear()
  original_output=native.lib.nm_output
  def output_fault(h,rows,ptr):
   if h==hg:return -1
   time.sleep(.02);rc=original_output(h,rows,ptr);done.set();return rc
  native.lib.nm_output=output_fault
  try:native.group([model.gate_proj,model.up_proj],x);raise AssertionError('Expected group failure')
  except RuntimeError:assert done.is_set();r['projection_group_joins_before_failure']=True
  finally:native.lib.nm_output=original_output
 # A down worker fails while its peer is still active: full CPU replay
 # must wait that peer before reusing or destroying either down buffer.
 native.lib.nm_raw=original_raw;precise.disabled.clear();precise.errors.clear();done.clear()
 h0=native.ensure_down(model.down_proj,1);h1=native.ensure_down(model.down_proj,2);saved_output=native.lib.nm_output
 def down_fault(h,rows,ptr):
  if h==h0:return -1
  time.sleep(.02);rc=saved_output(h,rows,ptr);done.set();return rc
 native.lib.nm_output=down_fault
 with torch.inference_mode():
  expected=cpu(x);out=model(x);assert done.is_set() and torch.equal(out,expected) and id(model) in precise.disabled;r['split_down_failure_joined_whole_cpu_fallback']=True
 native.lib.nm_output=saved_output
 precise.close();precise.close();native.close();native.close();assert not native.handles;r['double_close_empty']=True
 # Partial construction: first allocated handle remains owned and is freed.
 native=SplitMlp(engine,table);original_create=native.lib.nm_create;count={'n':0}
 def fail_second(*args):
  count['n']+=1
  return None if count['n']==2 else original_create(*args)
 native.lib.nm_create=fail_second
 try:native.run(members,x);raise AssertionError('Expected create failure')
 except RuntimeError:assert len(native.handles)==1
 native.close();assert not native.handles;r['partial_creation_cleanup']=True;r['completed']=True
finally:
 native.lib.nm_raw=original_raw;precise.close();native.close();engine.pool.shutdown(wait=True);r['handles_after_close']=len(native.handles);(p/'logs/split-guards.json').write_text(json.dumps(r,indent=2))
print('NATIVE GUARDS',r,flush=True)
