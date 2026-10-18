import sys
sys.path.insert(0,'/tail')
from runner import *
from precise_mlp import PreciseMlp
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ctypes,threading,time
p=Path('/experiment');r={'passed':False};done=threading.Event();fault={'mode':'range','prepares':0};events=[]
class Fn:
 def __init__(self,fn):self.fn=fn
 def __call__(self,*args):return self.fn(*args)
def init(self):
 torch.nn.Module.__init__(self);self.gate_proj=torch.nn.Linear(1024,8,bias=False,dtype=torch.bfloat16);self.up_proj=torch.nn.Linear(1024,8,bias=False,dtype=torch.bfloat16);self.down_proj=torch.nn.Linear(8,1024,bias=False,dtype=torch.bfloat16);self.act_fn=torch.nn.SiLU()
 for m in (self.gate_proj,self.up_proj,self.down_proj):
  w=m.weight.detach().float();m._bp_original=lambda x,w=w:torch.nn.functional.linear(x.float(),w).bfloat16()
def forward(self,x):return self.down_proj._bp_original(self.act_fn(self.gate_proj._bp_original(x))*self.up_proj._bp_original(x))
MLP=type('Qwen3_5MLP',(torch.nn.Module,),{'__init__':init,'forward':forward});model=MLP().eval();original=model.forward
class Engine:
 def __init__(self):
  self.mode='balanced';self.calls=0;self.pool=ThreadPoolExecutor(2)
  self.lib=type('Lib',(),{})()
  for name,fn in [('mm_prepare_bf16',lambda *a:0),('mm_copy_prepared',lambda *a:0),('mm_run_raw',self.raw),('mm_prepare_mlp',self.prepare),('mm_run_prepared_bf16',self.output)]:setattr(self.lib,name,Fn(fn))
 def ensure(self,m):return id(m)
 def status(self,rc):
  if rc:raise RuntimeError('injected native fault')
 def raw(self,h,rows):
  if h==id(model.up_proj):time.sleep(.02);done.set()
  if fault['mode']=='runtime' and h==id(model.gate_proj):return -1
  return 0
 def prepare(self,*args):
  fault['prepares']+=1
  return -2 if fault['mode']=='range' and fault['prepares']==2 else 0
 def output(self,handle,rows,bias,ptr):
  bad=torch.full((rows,1024),17,dtype=torch.bfloat16);ctypes.memmove(ptr,bad.data_ptr(),bad.numel()*2);return 0
engine=Engine();table=type('Table',(),{'silu':torch.zeros(65536,dtype=torch.bfloat16)})();precise=PreciseMlp(model,engine,table);precise.enabled=True
try:
 with torch.inference_mode():
  x=torch.randn(3,239,1024).bfloat16();expected=original(x);out=model(x);assert torch.equal(out,expected) and precise.range_fallbacks==1 and not precise.disabled;r['range_second_chunk_discards_private_output']=True
  long=torch.randn(2,611,1024).bfloat16();before=fault['prepares'];assert torch.equal(model(long),original(long)) and fault['prepares']==before;r['long_sequence_bypass']=True
  fault.update(mode='runtime',prepares=0);done.clear();out=model(x);assert done.is_set() and torch.equal(out,expected) and id(model) in precise.disabled;r['runtime_joins_disables_and_whole_cpu_fallback']=True
 precise.close();precise.close();assert model.forward==original
finally:precise.close();engine.pool.shutdown(wait=True)
r['passed']=True;(p/'logs/precise-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('PRECISE UNIT PASS',r,flush=True)
