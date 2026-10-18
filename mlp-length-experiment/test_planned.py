from test_buckets import BucketFake,Fake,Qwen3_5MLP
from planned_mlp import PlannedBank
from types import SimpleNamespace
from pathlib import Path
import torch,json
Fake.owners=[];Fake.fail_at=None;Fake.fail_call=None;BucketFake.made=[];BucketFake.fail_at=None
m=torch.nn.Sequential(*(Qwen3_5MLP() for _ in range(12))).bfloat16().eval();original=[x.forward for x in m];b=PlannedBank(SimpleNamespace(backbone=m),factory=BucketFake,base_factory=Fake);b.mode='planned';tested=[]
try:
 with torch.inference_mode():
  for batch,t in [(1,t) for t in [1,105,114,115,116,117,119,128,129,167,231,342,351,359,360,381,384,385,503,512,513,2048]]+[(3,127),(2,193)]:
   x=torch.arange(batch*t*1024,dtype=torch.float32).remainder(23).reshape(batch,t,1024).bfloat16();snap=x.clone();b.reset_counts();out=m[0](x);assert torch.equal(out,x*2) and torch.equal(x,snap),(batch,t,b.counts);assert b.counts['optimized']==1;tested.append([batch,t])
   plan=b.plans[batch*t];assert sum(plan)>=batch*t and sum(plan[:-1])<batch*t
  x=torch.ones(1,503,1024,dtype=torch.bfloat16);Fake.fail_call=Fake.calls+2;assert torch.equal(m[0](x),x+2);assert b.counts['range_fallback']==1;Fake.fail_call=None
finally:b.close();b.close()
assert all(v.forward==f for v,f in zip(m,original));assert all(v.closed==1 for v in Fake.owners+BucketFake.made)
r={'passed':True,'shapes':tested,'whole_layer_second_piece_fallback':True,'cost_calibration_and_plan_coverage':True,'calibration':b.calibration};Path('/experiment/logs/planned-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('PLANNED UNIT PASS',len(tested))
