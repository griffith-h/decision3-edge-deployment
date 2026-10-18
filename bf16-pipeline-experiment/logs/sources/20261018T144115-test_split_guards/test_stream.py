"""Segmented internal scan must equal original 32/64 output bit for bit."""
import sys
sys.path[:0]=['/tail','/long','/row','/mlp','/work','/remainder']
import ctypes,json,time,torch
from pathlib import Path
import long_kernels as lk
import tail_native as native
p=Path('/experiment');lib=ctypes.CDLL(str(p/'libstream_scan.so'));readytype=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p)
for name,fn in [('d3_set_sgemm',lk.blas.scipy_cblas_sgemm64_),('d3_set_strsm',lk.blas.scipy_cblas_strsm64_)]:getattr(lib,name).argtypes=[ctypes.c_void_p];getattr(lib,name)(ctypes.cast(fn,ctypes.c_void_p))
lib.d3_stream_scan.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*8+[readytype,ctypes.c_void_p];lib.d3_stream_scan.restype=ctypes.c_int
r={'cases':[],'completed':False};torch.set_num_threads(3);torch.manual_seed(17)
with torch.inference_mode():
 for length in (32,33,128,129,615,640):
  q=torch.randn(1,16,length,128);k=torch.randn_like(q);v=torch.randn_like(q);q=torch.nn.functional.normalize(q,dim=-1)/128**.5;k=torch.nn.functional.normalize(k,dim=-1);decay=torch.rand(1,16,length)*.5+.5;beta=torch.rand(1,16,length);ref=native.scan(q,k,v,decay,beta,kind='original')
  for group in (128,256,640):
   out=torch.empty_like(v,dtype=torch.bfloat16);seen=[]
   @readytype
   def ready(start,rows,user):
    seen.append((start,rows));assert torch.equal(out[:,:,start:start+rows],ref[:,:,start:start+rows]);return 0
   t=time.perf_counter();rc=lib.d3_stream_scan(*(x.data_ptr() for x in (q,k,v,decay,beta,out)),16,length,128,128,32,64,group,4,ready,None);seconds=time.perf_counter()-t
   assert rc==0 and torch.equal(out,ref);r['cases'].append({'length':length,'group':group,'ready':seen,'seconds':seconds,'bit_equal':True})
 r['completed']=True
(p/'logs/stream-unit.json').write_text(json.dumps(r,indent=2));print('STREAM UNIT',len(r['cases']),flush=True)
