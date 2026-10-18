import sys
sys.path.insert(0,'/tail')
from runner import *
import ctypes
p=Path('/experiment');lib=ctypes.CDLL(str(p/'libneon_scan.so'))
for name in ('d3_set_sgemm','d3_set_strsm','d3_chunk_scan'):
 a=getattr(lk.scan,name);b=getattr(lib,name);b.argtypes=a.argtypes;b.restype=a.restype
lib.d3_set_sgemm(ctypes.cast(lk.blas.scipy_cblas_sgemm64_,ctypes.c_void_p));lib.d3_set_strsm(ctypes.cast(lk.blas.scipy_cblas_strsm64_,ctypes.c_void_p));r={'records':[],'passed':False};torch.manual_seed(20261018)
with torch.inference_mode():
 for length in (1,31,32,33,511,512,513,615,2048):
  q=torch.nn.functional.normalize(torch.randn(1,2,length,128),dim=-1);k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1);v=torch.randn_like(q);beta=torch.rand(1,2,length);decay=.8+.2*torch.rand(1,2,length)
  for boundary in (False,True):
   if boundary:beta.fill_(1);decay.fill_(1)
   outputs=[]
   for library in (lk.scan,lib):
    out=torch.empty_like(v,dtype=torch.bfloat16);args=[x.data_ptr() for x in (q,k,v,decay,beta,out)];rc=library.d3_chunk_scan(*args,2,length,128,128,32,64);assert rc==0;outputs.append(out)
   delta=(outputs[0].float()-outputs[1].float()).abs();ok=torch.allclose(outputs[0].float(),outputs[1].float(),atol=.002,rtol=.008);r['records'].append({'length':length,'beta_decay_one':boundary,'max_hidden_delta':float(delta.max()),'allclose':ok});assert ok,r['records'][-1]
 r['passed']=True
(p/'logs/neon-scan-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('NEON SCAN UNIT PASS',r,flush=True)
