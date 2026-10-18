import sys
sys.path.insert(0,'/tail')
from runner import *
import ctypes,statistics
p=Path('/experiment');lib=ctypes.CDLL(str(p/'libfused_scan.so'))
for name in ('d3_set_sgemm','d3_set_strsm','d3_chunk_scan'):
 a=getattr(lk.scan,name);b=getattr(lib,name);b.argtypes=a.argtypes;b.restype=a.restype
lib.d3_set_sgemm(ctypes.cast(lk.blas.scipy_cblas_sgemm64_,ctypes.c_void_p));lib.d3_set_strsm(ctypes.cast(lk.blas.scipy_cblas_strsm64_,ctypes.c_void_p))
r={'settings':state(),'records':[],'scope':'Fuses Q/K GEMM submissions and shares packed input; original block32/tile64 and FP32 cross-block state maintained. Component replay only.'};captured=torch.load('/tail/captured.pt',weights_only=False)['scan']
with torch.inference_mode():
 for rep in range(5):
  for i,item in enumerate(captured):
   q,k,v,decay,beta=item['args'];out=torch.empty(v.shape,dtype=torch.bfloat16);args=[x.data_ptr() for x in (q,k,v,decay,beta,out)];shape=(q.shape[0]*q.shape[1],q.shape[2],q.shape[3],v.shape[3])
   for mode in (('baseline','fused') if rep%2==0 else ('fused','baseline')):
    fn=lk.scan.d3_chunk_scan if mode=='baseline' else lib.d3_chunk_scan;t=time.perf_counter();rc=fn(*args,*shape,32,64);dt=time.perf_counter()-t;delta=(out.float()-item['reference'].float()).abs();assert rc==0
    rec={'mode':mode,'rep':rep,'layer':i,'seconds':dt,'max_hidden_delta':float(delta.max()),'allclose':torch.allclose(out.float(),item['reference'].float(),atol=.002,rtol=.008)};r['records'].append(rec);print('FUSED SCAN',rec,flush=True)
r['completed']=True;(p/'logs/replay-fused-scan.json').write_text(json.dumps(r,indent=2)+'\n')
