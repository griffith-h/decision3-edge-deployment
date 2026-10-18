from test_rows import Qwen3_5MLP,Fake
from bucket_mlp import BucketBank
from types import SimpleNamespace
import json,torch,re
from pathlib import Path
class BucketFake(Fake):
 made=[];fail_at=None
 def __init__(self,path):
  if len(self.made)==self.fail_at:raise RuntimeError('extra ctor failure')
  self.rows=int(re.search(r'rows(\d+)',str(path))[1]);self.closed=0;self.made.append(self)
 def run_bf16(self,x):
  assert x.shape==(1,self.rows,1024)
  return super().run_bf16(x)
Fake.owners=[];Fake.fail_at=None;Fake.fail_call=None
m=torch.nn.Sequential(*(Qwen3_5MLP() for _ in range(12))).bfloat16().eval();old=[x.forward for x in m];b=BucketBank(SimpleNamespace(backbone=m),factory=BucketFake,base_factory=Fake);b.mode='bucket';tested=[]
try:
 with torch.inference_mode():
  for batch,t in [(1,t) for t in [1,114,115,116,117,127,128,129,383,384,385,511,512,513,767,768,769,2048]]+[(3,127),(3,128),(2,193)]:
   x=torch.arange(batch*t*1024,dtype=torch.float32).remainder(19).reshape(batch,t,1024).bfloat16();snap=x.clone();b.reset_counts();out=m[0](x);assert torch.equal(out,x*2) and torch.equal(x,snap),(batch,t,b.counts);assert b.counts['optimized']==1;tested.append([batch,t])
  x=torch.zeros(1,385,1024,dtype=torch.bfloat16);Fake.fail_call=Fake.calls+2;assert torch.equal(m[0](x),x+2);assert b.counts['range_fallback']==1;Fake.fail_call=None
finally:b.close();b.close()
assert all(v.forward==f for v,f in zip(m,old));assert all(v.closed==1 for v in Fake.owners+BucketFake.made)
Fake.owners=[];BucketFake.made=[];BucketFake.fail_at=4
try:BucketBank(SimpleNamespace(backbone=m),factory=BucketFake,base_factory=Fake);raise AssertionError('ctor did not fail')
except RuntimeError:pass
assert all(v.forward==f for v,f in zip(m,old));assert all(v.closed==1 for v in Fake.owners+BucketFake.made)
r={'passed':True,'shapes':tested,'extra_constructor_cleanup':True,'whole_mlp_on_partial_tile_failure':True};Path('/experiment/logs/bucket-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('BUCKET UNIT PASS',len(tested))
