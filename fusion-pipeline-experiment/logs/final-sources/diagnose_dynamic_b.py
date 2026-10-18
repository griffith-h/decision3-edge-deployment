import sys
sys.path.insert(0,'/tail')
from runner import *
import ctypes
p=Path('/experiment');lib=ctypes.CDLL(str(p/'libdynamic_matmul.so'));lib.dm_error.restype=ctypes.c_char_p;lib.dm_create.argtypes=[ctypes.c_int]*4;lib.dm_create.restype=ctypes.c_void_p;lib.dm_set_b.argtypes=[ctypes.c_void_p]*2;lib.dm_run.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int];lib.dm_destroy.argtypes=[ctypes.c_void_p]
r={'records':[]};handles=[]
def check(rc):
 if rc:raise RuntimeError(lib.dm_error().decode())
try:
 torch.manual_seed(20261018)
 with torch.inference_mode():
  for m,k,n in [(128,256,624),(615,256,624),(615,640,256),(640,256,640)]:
   h=lib.dm_create(m,k,n,1)
   if not h:raise RuntimeError(lib.dm_error().decode())
   handles.append(h)
   for rep in range(4):
    a=(torch.randn(m,k)*.1).half();b=(torch.randn(k,n)*.1).half();reference=a.float()@b.float();out=torch.empty(m,n)
    check(lib.dm_set_b(h,b.data_ptr()));check(lib.dm_run(h,a.data_ptr(),out.data_ptr(),m));again=torch.empty_like(out);check(lib.dm_run(h,a.data_ptr(),again.data_ptr(),m))
    diff=(out-reference).abs();rec={'shape':[m,k,n],'rep':rep,'max':float(diff.max()),'mean':float(diff.mean()),'repeat_delta':float((again-out).abs().max()),'bad_count':int((diff>.001).sum())};r['records'].append(rec);print('DYNAMIC',rec,flush=True)
 r['completed']=True
finally:
 for h in reversed(handles):lib.dm_destroy(h)
 r['destroyed']=len(handles);(p/'logs/diagnose-dynamic-b.json').write_text(json.dumps(r,indent=2)+'\n')
