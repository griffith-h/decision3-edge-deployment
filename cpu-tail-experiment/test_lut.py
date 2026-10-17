import ctypes,json,time,torch
from pathlib import Path
from lut_ops import Tables,activation,lib
from baseline import stats
from types import SimpleNamespace
from lut_ops import install
import torch.nn.functional as F
r={'checks':[]};torch.set_num_threads(3)
def bitsame(a,b):
 finite=torch.isfinite(b);assert torch.equal(torch.isnan(a),torch.isnan(b));assert torch.equal(torch.isinf(a),torch.isinf(b));assert torch.equal(a[finite].view(torch.int16),b[finite].view(torch.int16))
with torch.inference_mode():
 t=Tables();values=torch.arange(65536,dtype=torch.int32).short().view(torch.bfloat16)
 for sig in (False,True):
  out=activation(values,t,sigmoid=sig);ref=torch.sigmoid(values) if sig else F.silu(values);bitsame(out,ref);r['checks'].append('all65536 sigmoid='+str(sig))
 torch.manual_seed(17)
 for n in (1,3,4,5,17,2047,2048,2049,65536,470927):
  x=(torch.randn(n)*8).bfloat16();u=(torch.randn(n)*3).bfloat16();saved=(x.clone(),u.clone())
  bitsame(activation(x,t,u),F.silu(x)*u);assert torch.equal(x,saved[0]) and torch.equal(u,saved[1]);r['checks'].append('mul n='+str(n))
 for kind in (0,1):
  for dim in (1,3,4,5,128,256,1024):
   rows=17;inp=torch.randn(rows,dim).bfloat16();w=torch.randn(dim).bfloat16();gate=torch.randn_like(inp);out=torch.empty_like(inp)
   lib.d3_norm_lut(inp.data_ptr(),w.data_ptr(),gate.data_ptr() if kind else None,out.data_ptr(),t.gate.data_ptr(),rows,dim,1e-6,kind)
   # Same four-lane accumulation order as frozen native baseline, in FP32.
   acc=torch.zeros(rows,4)
   for i in range(0,dim-dim%4,4):acc+=inp[:,i:i+4].float().square()
   # AArch64 vaddvq sums adjacent pairs.
   s=(acc[:,0]+acc[:,1])+(acc[:,2]+acc[:,3])
   for i in range(dim-dim%4,dim):s+=inp[:,i].float().square()
   scale=(s/dim+1e-6).sqrt().reciprocal();v=inp.float()*scale[:,None]
   if kind:
    v=(v.bfloat16()*w).bfloat16().float();g=t.gate[gate.view(torch.int16).int()&65535];v*=g
   else:v*=1+w.float()
   bitsame(out,v.bfloat16());r['checks'].append('norm '+str((kind,dim)))
 # Fallback dispatch preserves grad, inplace and out semantics and restores globals.
 old=F.silu;old_sig=torch.sigmoid;model=SimpleNamespace(backbone=torch.nn.Identity());restore,c=install(model,None,t,'activation')
 x=torch.randn(9).bfloat16();assert F.silu(x,inplace=True) is x
 y=torch.empty_like(x);assert torch.sigmoid(x,out=y) is y
 with torch.inference_mode(False), torch.enable_grad():
  z=torch.randn(3,requires_grad=True);F.silu(z).sum().backward();assert z.grad is not None
 restore();assert F.silu is old and torch.sigmoid is old_sig;r['fallback_counts']=c
 r['passed']=True
Path('/work/logs/lut-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('UNIT',len(r['checks']),r['passed'],flush=True)
