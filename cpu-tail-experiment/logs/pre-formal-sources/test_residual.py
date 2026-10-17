import torch,ctypes,json
from pathlib import Path
from residual import add_norm
from types import SimpleNamespace
r={'cases':[]};torch.set_num_threads(3);torch.manual_seed(122)
with torch.inference_mode():
 for rows in (1,3,17,115):
  for d in (1,3,4,5,128,1024):
   x=torch.randn(rows,d).bfloat16();y=torch.randn_like(x);w=torch.randn(d).bfloat16();n=SimpleNamespace(weight=w,eps=1e-6)
   z,o=add_norm(x,y,n);expected=x+y;assert torch.equal(z,expected)
   zz,_=add_norm(x,y);assert torch.equal(zz,expected)
   # Same frozen norm oracle reduction, checked independently in test_lut.py.
   acc=torch.zeros(rows,4)
   for i in range(0,d-d%4,4):acc+=expected[:,i:i+4].float().square()
   s=(acc[:,0]+acc[:,1])+(acc[:,2]+acc[:,3])
   for i in range(d-d%4,d):s+=expected[:,i].float().square()
   ref=((expected.float()*((s/d+1e-6).sqrt().reciprocal())[:,None])*(1+w.float())).bfloat16()
   assert torch.equal(o,ref),(rows,d)
   r['cases'].append([rows,d])
 r['passed']=True
Path('/work/logs/residual-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('RESIDUAL UNIT',len(r['cases']),flush=True)
