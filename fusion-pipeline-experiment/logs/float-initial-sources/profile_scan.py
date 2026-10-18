import sys
sys.path.insert(0,'/tail')
from runner import *
import ctypes
p=Path('/experiment');lib=ctypes.CDLL(str(p/'libprofile_scan.so'))
for name in ('d3_set_sgemm','d3_set_strsm'):
 old=getattr(lk.scan,name);fn=getattr(lib,name);fn.argtypes=old.argtypes;fn.restype=old.restype
lib.d3_chunk_scan.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*6+[ctypes.c_void_p];lib.d3_chunk_scan.restype=ctypes.c_int
lib.d3_set_sgemm(ctypes.cast(lk.blas.scipy_cblas_sgemm64_,ctypes.c_void_p));lib.d3_set_strsm(ctypes.cast(lk.blas.scipy_cblas_strsm64_,ctypes.c_void_p));captured=torch.load('/tail/captured.pt',weights_only=False)['scan'];r={'settings':state(),'records':[],'scope':'Instrumented component replay; stage values are worker time sums, not elapsed wall time'}
with torch.inference_mode():
 for rep in range(3):
  for i,item in enumerate(captured):
   q,k,v,decay,beta=item['args'];out=torch.empty(v.shape,dtype=torch.bfloat16);args=[x.data_ptr() for x in (q,k,v,decay,beta,out)];h=q.shape[0]*q.shape[1];shape=(h,q.shape[2],q.shape[3],v.shape[3]);tiles=(v.shape[3]+63)//64;values=(ctypes.c_double*(h+h*tiles*5))()
   t=time.perf_counter();rc=lib.d3_chunk_scan(*args,*shape,32,64,ctypes.addressof(values));dt=time.perf_counter()-t;assert rc==0 and torch.equal(out,item['reference'])
   stages={'coefficients':sum(values[:h]),**{name:sum(values[h+job*5+j] for job in range(h*tiles)) for j,name in enumerate(('predictions','vector_prepare','triangular_solve','readout_write','state_update'))}}
   rec={'rep':rep,'layer':i,'seconds':dt,'worker_seconds':stages};r['records'].append(rec);print('SCAN PROFILE',rec,flush=True)
r['completed']=True;(p/'logs/profile-scan.json').write_text(json.dumps(r,indent=2)+'\n')
