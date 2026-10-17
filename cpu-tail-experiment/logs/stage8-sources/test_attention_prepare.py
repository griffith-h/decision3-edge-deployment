import torch,json,ctypes
from baseline import load_baseline
import native_norm
from pathlib import Path
from types import SimpleNamespace
from attention_prepare import prepare,gated
from lut_ops import Tables
r={'cases':[]};torch.set_num_threads(3);torch.manual_seed(332)
oracle=ctypes.CDLL(str(Path(native_norm.__file__).with_name('libd3_native_norm.so')))
oracle.d3_norm.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_int];oracle.d3_norm.restype=None
def norm(x,w,eps):
 x=x.contiguous();w=w.contiguous();out=torch.empty_like(x)
 oracle.d3_norm(x.data_ptr(),w.data_ptr(),None,out.data_ptr(),x.numel()//x.shape[-1],x.shape[-1],eps,0)
 return out
def rope(q,k,c,s):
 c=c.unsqueeze(1);s=s.unsqueeze(1);rot=c.shape[-1]
 def f(x):
  a=x[...,:rot];l,h=a.chunk(2,-1);return torch.cat((a*c+torch.cat((-h,l),-1)*s,x[...,rot:]),-1)
 return f(q),f(k)
with torch.inference_mode():
 table=Tables()
 for b in (1,2):
  for t in (1,3,17,115):
   for d,rot in ((32,16),(128,32),(256,64),(256,256)):
    h,kh=8,2;qr=torch.randn(b,t,h*2*d).bfloat16();kr=torch.randn(b,t,kh*d).bfloat16();qw=torch.randn(d).bfloat16();kw=torch.randn(d).bfloat16();c=torch.randn(b,t,rot).bfloat16();s=torch.randn_like(c)
    q=qr.view(b,t,h,2*d)[...,:d].contiguous();k=kr.view(b,t,kh,d)
    expected=rope(norm(q,qw,1e-6).transpose(1,2),norm(k,kw,1e-6).transpose(1,2),c,s);actual=prepare(qr,kr,qw,kw,c,s,1e-6,1e-6,h,kh,d)
    assert all(torch.equal(a,z) for a,z in zip(actual,expected)),(b,t,d,rot)
    x=torch.randn(b,t,h,d).bfloat16();ref=x*torch.sigmoid(qr.view(b,t,h,2*d)[...,d:]);assert torch.equal(gated(x,qr,table),ref.flatten(-2))
    r['cases'].append([b,t,d,rot])
 r['passed']=True
Path('/work/logs/attention-prepare-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('ATTENTION PREP UNIT',len(r['cases']),flush=True)
