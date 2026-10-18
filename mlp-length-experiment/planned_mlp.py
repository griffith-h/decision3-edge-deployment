"""Choose among existing 115/128/384 graphs using measured graph costs, no answer cache."""
from bucket_mlp import BucketBank
import torch,time,statistics
class PlannedBank(BucketBank):
 def __init__(self,model,**kwargs):
  super().__init__(model,**kwargs)
  try:
   samples={n:[] for n in (115,128,384)}
   with torch.inference_mode():
    for i in (0,5,11):
     base=self.graphs[i]
     for n in samples:
      g=base if n==115 else self.by_id[id(base)][n];x=torch.zeros(1,n,1024,dtype=torch.bfloat16)
      for _ in range(3):g.run_bf16(x)
      for _ in range(15):
       t=time.perf_counter();g.run_bf16(x);samples[n].append(time.perf_counter()-t)
   self.costs={n:statistics.median(v) for n,v in samples.items()};self.calibration={'costs':self.costs,'samples':samples};self.plans={};dp=[(0.,())]
   for rows in range(1,self.max_rows+1):
    cost,plan=min((self.costs[n]+dp[max(0,rows-n)][0],(n,)+dp[max(0,rows-n)][1]) for n in samples)
    dp.append((cost,plan));self.plans[rows]=plan
  except BaseException:self.close();raise
 def run_rows(self,g,x):
  n=x.shape[0]*x.shape[1]
  if self.mode!='planned' or n==115:return super().run_rows(g,x)
  plan=self.plans[n];key=','.join(map(str,plan));self.counts.setdefault('plans',{});self.counts['plans'][key]=self.counts['plans'].get(key,0)+1
  flat=x.reshape(-1,1024);out=torch.empty_like(flat);start=0
  for width in plan:
   used=min(width,n-start);h=g if width==115 else self.by_id[id(g)][width]
   value=self.run_packed(h,flat[start:start+used].reshape(1,used,1024),width);out[start:start+used].copy_(value.reshape(used,1024));start+=used
  assert start==n
  return out.reshape(x.shape)
