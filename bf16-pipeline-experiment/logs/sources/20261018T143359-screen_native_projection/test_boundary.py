"""Exact BF16 boundary fusion vs frozen own implementation, including guards."""
import ctypes,json,time,statistics,torch
from pathlib import Path
p=Path('/experiment');torch.set_num_threads(3);torch.manual_seed(42);lib=ctypes.CDLL(str(p/'libfused_boundary.so'));lib.mm_probe_boundary.argtypes=[ctypes.c_void_p]*4+[ctypes.c_size_t,ctypes.c_int];lib.mm_probe_boundary.restype=ctypes.c_int
allbf=torch.arange(65536,dtype=torch.int32).to(torch.uint16).view(torch.bfloat16);table=torch.nn.functional.silu(allbf);r={'cases':[],'timings':[],'completed':False}
def run(g,u,n):
 outputs=[];codes=[]
 for fused in (0,1):
  out=torch.empty(n,dtype=torch.float16);rc=lib.mm_probe_boundary(g.data_ptr(),u.data_ptr(),out.data_ptr(),table.data_ptr(),n,fused);outputs.append(out);codes.append(rc)
 assert codes[0]==codes[1],codes
 if codes[0]==0:assert torch.equal(outputs[0].view(torch.uint16),outputs[1].view(torch.uint16))
 return codes[0]
with torch.inference_mode():
 for n in (1,7,8,31,128,2047,2048,2049,615*3072):
  g=torch.randn(n)*3;u=torch.randn(n)*2;r['cases'].append({'n':n,'code':run(g,u,n),'equal':True})
 for kind,value in [('nan',float('nan')),('inf',float('inf')),('saturation',65496.),('product-overflow',60000.),('denormal',1.e-30),('negative-zero',-0.)]:
  g=torch.full((17,),value);u=torch.full((17,),value);code=run(g,u,17);r['cases'].append({'case':kind,'code':code,'equal':True})
  if kind in ('nan','inf','saturation'):assert code==-3
  if kind=='product-overflow':assert code==-2
 n=615*3072;g=torch.randn(n);u=torch.randn(n);out=torch.empty(n,dtype=torch.float16)
 for _ in range(3):run(g,u,n)
 for rep in range(20):
  for fused in ((0,1) if rep%2==0 else (1,0)):
   t=time.perf_counter();rc=lib.mm_probe_boundary(g.data_ptr(),u.data_ptr(),out.data_ptr(),table.data_ptr(),n,fused);dt=time.perf_counter()-t;assert rc==0;r['timings'].append({'pair':rep,'fused':fused,'seconds':dt})
 r['completed']=True;r['median']={str(v):statistics.median(x['seconds'] for x in r['timings'] if x['fused']==v) for v in (0,1)}
(p/'logs/boundary-unit.json').write_text(json.dumps(r,indent=2));print('BOUNDARY UNIT',r['median'],flush=True)
