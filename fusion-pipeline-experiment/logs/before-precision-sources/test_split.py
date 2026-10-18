import sys
sys.path.insert(0,'/tail')
from runner import *
from split_projection import SplitProjection
from safe_engine import OutputRangeError
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading,ctypes,time
p=Path('/experiment');r={'passed':False};done=threading.Event();fault={'kind':None};calls=[]
class Lib:
 def mm_prepare_bf16(self,*args):return 0
 def mm_copy_prepared(self,*args):return 0
class Engine:
 def __init__(self):self.execute=self.original;self.pool=ThreadPoolExecutor(2);self.lib=Lib();self.core={};self.bucket={};self.handles=[]
 def original(self,m,x,prepared=False,output=None):calls.append('fallback');return torch.zeros_like(x)
 def ensure(self,m):return id(m)
 def status(self,rc):assert rc==0
 def plan(self,modules,parts):return [0,1]
 def output_buffer(self,m,x):return torch.empty(*x.shape[:-1],m.out_features,dtype=x.dtype)
 @torch.inference_mode()
 def run_piece(self,m,x,outputs,plan):
  out=next(iter(outputs.values()));width=m.out_features
  if plan==1:time.sleep(.02);out[...,width:]=2;done.set()
  else:
   if fault['kind']=='range':raise OutputRangeError('injected range')
   if fault['kind']=='runtime':raise RuntimeError('injected failure')
   out[...,:width]=1
m=torch.nn.Linear(1024,1024,bias=False,dtype=torch.bfloat16);m.eval()
with torch.inference_mode():
 projections=SimpleNamespace(modules={'language_model.layers.0.self_attn.o_proj':m});engine=Engine();original=engine.execute;split=SplitProjection(projections,engine);split.enabled=True;x=torch.randn(1,103,1024).bfloat16()
 try:
  out=engine.execute(m,x);assert torch.all(out[...,:512]==1) and torch.all(out[...,512:]==2)
  for kind,exception in [('range',ValueError),('runtime',RuntimeError)]:
   done.clear();fault['kind']=kind
   try:engine.execute(m,x);raise AssertionError('fault ignored')
   except exception:assert done.is_set(),'did not join other worker'
  fault['kind']=None
  m.weight.add_(1);out=engine.execute(m,x);assert len(calls)==1 and not torch.count_nonzero(out)
  split.close();split.close();assert engine.execute==original
 finally:split.close();engine.pool.shutdown(wait=True)
r.update(passed=True,column_order=True,joined_before_range_or_runtime_error=True,weight_version_guard=True,restore=True);(p/'logs/split-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('SPLIT UNIT PASS',r,flush=True)
