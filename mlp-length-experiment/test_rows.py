"""Focused packing/ownership tests; hardware parity is tested separately."""
from types import SimpleNamespace
import json,torch
from pathlib import Path
from row_mlp import RowBank
class Qwen3_5MLP(torch.nn.Module):
 def __init__(self):
  super().__init__();self.gate_proj=torch.nn.Linear(1,1,bias=False);self.up_proj=torch.nn.Linear(1,1,bias=False);self.down_proj=torch.nn.Linear(1,1,bias=False);self.act_fn=torch.nn.SiLU()
 def forward(self,x):return x+2
class Fake:
 owners=[];fail_at=None;fail_call=None;calls=0;fail_type=ValueError
 def __init__(self,path):
  if len(self.owners)==self.fail_at:raise RuntimeError('ctor failure')
  self.closed=0;self.owners.append(self)
 def run_bf16(self,x):
  Fake.calls+=1
  if Fake.calls==Fake.fail_call:raise Fake.fail_type('injected failure')
  return x*2
 def close(self):self.closed+=1
m=torch.nn.Sequential(*(Qwen3_5MLP() for _ in range(12))).bfloat16().eval();old=[x.forward for x in m];b=RowBank(SimpleNamespace(backbone=m),Fake);b.mode='tile115';tested=[]
try:
 with torch.inference_mode():
  for batch,t in [(1,t) for t in [1,64,114,115,116,117,229,230,231,314,344,345,346,512,513,1024,2048]]+[(2,117),(3,115)]:
   x=torch.arange(batch*t*1024,dtype=torch.float32).remainder(23).reshape(batch,t,1024).bfloat16();snap=x.clone();out=m[0](x);assert torch.equal(out,x*2) and torch.equal(x,snap);tested.append([batch,t])
  x=torch.arange(116*2048,dtype=torch.float32).remainder(19).reshape(1,116,2048).bfloat16()[...,::2];assert not x.is_contiguous();assert torch.equal(m[0](x),x*2)
  # Second tile fails: entire original MLP result, never mixed with first tile.
  Fake.fail_call=Fake.calls+2;assert torch.equal(m[0](x),x+2);Fake.fail_call=None
  Fake.fail_call=Fake.calls+2;Fake.fail_type=RuntimeError;assert torch.equal(m[4](x),x+2) and 4 in b.disabled;Fake.fail_call=None;Fake.fail_type=ValueError;assert torch.equal(m[4](x),x+2)
  x=torch.zeros(1,2049,1024,dtype=torch.bfloat16);assert torch.equal(m[0](x),x+2)
  x=torch.zeros(1,116,1024,dtype=torch.bfloat16);m[0].gate_proj.weight.add_(1);assert torch.equal(m[0](x),x+2)
  m[1].train();assert torch.equal(m[1](x),x+2);m[1].eval()
  h=m[2].gate_proj.register_forward_pre_hook(lambda *a:None);assert torch.equal(m[2](x),x+2);h.remove()
 x=torch.zeros(1,115,1024,dtype=torch.bfloat16,requires_grad=True);out=m[3](x);assert torch.equal(out,x+2) and out.requires_grad
finally:b.close()
assert all(v.forward==f for v,f in zip(m,old)) and all(v.closed==1 for v in Fake.owners)
Fake.owners=[];Fake.fail_at=4
try:RowBank(SimpleNamespace(backbone=m),Fake);raise AssertionError('ctor did not fail')
except RuntimeError:pass
assert all(v.forward==f for v,f in zip(m,old)) and all(v.closed==1 for v in Fake.owners)
r={'passed':True,'shapes':tested,'noncontiguous':True,'second_tile_whole_fallback':True,'guards_and_cleanup':True};Path('/experiment/logs/row-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('ROW UNIT PASS',len(tested))
