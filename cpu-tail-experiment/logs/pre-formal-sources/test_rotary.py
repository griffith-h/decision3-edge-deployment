import torch,json
from pathlib import Path
from rotary import rotary
r={'cases':[]};torch.set_num_threads(3);torch.manual_seed(159)
def ref(q,k,c,s):
 c=c.unsqueeze(1);s=s.unsqueeze(1);rot=c.shape[-1]
 def f(x):
  a=x[...,:rot];left,right=a.chunk(2,-1);half=torch.cat((-right,left),-1)
  return torch.cat((a*c+half*s,x[...,rot:]),-1)
 return f(q),f(k)
with torch.inference_mode():
 for b in (1,2):
  for t in (1,3,17,115):
   for d,rot in ((8,8),(32,16),(256,64),(256,256)):
    q=torch.randn(b,t,8,d).bfloat16().transpose(1,2);k=torch.randn(b,t,2,d).bfloat16().transpose(1,2);c=torch.randn(b,t,rot).bfloat16();s=torch.randn_like(c)
    expect=ref(q,k,c,s);before=[x.clone() for x in (q,k,c,s)];actual=rotary(q,k,c,s)
    assert all(torch.equal(x,y) for x,y in zip(expect,actual)),(b,t,d,rot)
    assert all(torch.equal(x,y) for x,y in zip((q,k,c,s),before));r['cases'].append([b,t,d,rot])
 r['passed']=True
Path('/work/logs/rotary-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('ROTARY UNIT',len(r['cases']),flush=True)
