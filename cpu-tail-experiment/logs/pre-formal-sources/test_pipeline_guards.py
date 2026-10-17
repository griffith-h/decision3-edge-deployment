from baseline import load_baseline
from pipeline import install
import torch,ctypes,time,json
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
class Qwen3_5MLP(torch.nn.Module):
 def __init__(self):
  super().__init__();self.gate_proj=torch.nn.Linear(16,32,bias=False).bfloat16();self.up_proj=torch.nn.Linear(16,32,bias=False).bfloat16();self.down_proj=torch.nn.Linear(32,16,bias=False).bfloat16();self.act_fn=torch.nn.SiLU();self.old_calls=0
 def forward(self,x):self.old_calls+=1;return torch.ones(*x.shape[:-1],16,dtype=x.dtype)
class Fn:
 def __init__(self,f):self.f=f
 def __call__(self,*args):return self.f(*args)
m=Qwen3_5MLP().eval()
for p in (m.gate_proj,m.up_proj,m.down_proj):p._bp_original=p.forward
state={'prepare':0,'down':0,'error':False,'done':0}
def raw(handle,rows):
 try:
  time.sleep(.005)
  if state['error'] and handle==id(m.gate_proj):raise RuntimeError('injected worker error')
  return 0
 finally:state['done']+=1
def down(handle,rows,bias,out):
 if not state['down']:ctypes.memset(out,0,rows*16*2)
 return state['down']
lib=SimpleNamespace(mm_run_raw=Fn(raw),mm_prepare_mlp=Fn(lambda *a:state['prepare']),mm_run_prepared_bf16=Fn(down),mm_prepare_bf16=Fn(lambda *a:0),mm_copy_prepared=Fn(lambda *a:0))
engine=SimpleNamespace(mode='balanced',pool=ThreadPoolExecutor(max_workers=2),lib=lib,pack=lambda ms:None,eligible=lambda m,x:True,ensure=lambda m:id(m),status=lambda c:None,calls=0)
model=SimpleNamespace(backbone=m);table=SimpleNamespace(silu=torch.zeros(65536,dtype=torch.bfloat16));original=m.forward;restore,c=install(model,engine,table);r={}
try:
 with torch.inference_mode():
  x=torch.zeros(1,3,16,dtype=torch.bfloat16);out=m(x);assert torch.count_nonzero(out)==0 and c['optimized']==1
  state['prepare']=-3;assert bool((m(x)==1).all());state['prepare']=0
  state['down']=-3;assert bool((m(x)==1).all());state['down']=0
  assert c['range_fallback']==2 and m.old_calls==2
  state['error']=True;before=state['done']
  try:m(x);raise AssertionError('missing worker exception')
  except RuntimeError as e:assert str(e)=='injected worker error'
  assert state['done']-before==2
  state['error']=False
 with torch.enable_grad():m(torch.zeros(1,3,16,dtype=torch.bfloat16));assert c['fallback']>0
 restore();assert m.forward==original;r={'passed':True,'counts':c,'all_workers_joined':True,'down_overflow_falls_back_whole_mlp':True}
finally:engine.pool.shutdown(wait=True)
Path('/work/logs/pipeline-guards.json').write_text(json.dumps(r,indent=2)+'\n');print('PIPELINE GUARDS PASS',flush=True)
