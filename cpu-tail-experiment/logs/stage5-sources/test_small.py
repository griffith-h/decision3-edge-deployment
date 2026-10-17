import torch,json
from types import SimpleNamespace
from pathlib import Path
from small_linear import install
r={'cases':[]};torch.set_num_threads(3);torch.manual_seed(291)
for merge in (False,True):
 class Qwen3_5GatedDeltaNet(torch.nn.Module):
  def __init__(self):
   super().__init__();self.in_proj_a=torch.nn.Linear(32,16,bias=False).bfloat16();self.in_proj_b=torch.nn.Linear(32,16,bias=False).bfloat16()
  def forward(self,x):return self.in_proj_a(x),self.in_proj_b(x)
 m=Qwen3_5GatedDeltaNet().eval();model=SimpleNamespace(backbone=m);x=torch.randn(1,17,32).bfloat16()
 old=m.forward;restore,c=install(model,merge)
 with torch.inference_mode():
  for length in (1,3,17,115):
   z=torch.randn(1,length,32).bfloat16();ref=(torch.nn.functional.linear(z,m.in_proj_a.weight),torch.nn.functional.linear(z,m.in_proj_b.weight));out=m(z)
   for a,b in zip(out,ref):torch.testing.assert_close(a,b,atol=.008,rtol=.008)
   r['cases'].append({'merge':merge,'length':length,'delta':max((a.float()-b.float()).abs().max().item() for a,b in zip(out,ref))})
  # Changed weights must use the current model weights; stale merged cache cannot win.
  m.in_proj_a.weight.add_(1);out=m(x);expected=torch.nn.functional.linear(x,m.in_proj_a.weight);assert torch.equal(out[0],expected)
 with torch.enable_grad():
  y=torch.randn(1,3,32,dtype=torch.bfloat16,requires_grad=True);sum(v.float().sum() for v in m(y)).backward();assert y.grad is not None
 assert c['fallback']>0;restore();assert m.forward==old
r['passed']=True;Path('/work/logs/small-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('SMALL UNIT',len(r['cases']),flush=True)
