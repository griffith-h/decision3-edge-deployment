import sys
sys.path.insert(0,'/tail')
from runner import *
import ctypes,statistics
p=Path('/experiment');lib=ctypes.CDLL(str(p/'libdynamic_matmul.so'));lib.dm_error.restype=ctypes.c_char_p;lib.dm_create.argtypes=[ctypes.c_int]*4;lib.dm_create.restype=ctypes.c_void_p
lib.dm_set_b.argtypes=[ctypes.c_void_p]*2;lib.dm_run.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int];lib.dm_destroy.argtypes=[ctypes.c_void_p]
r={'settings':state(),'records':[],'scope':'Real-layer replay only, not full-model timing or output gate'};handles=[]
def check(rc):
 if rc:raise RuntimeError(lib.dm_error().decode())
def create(m,k,n):
 h=lib.dm_create(m,k,n,1)
 if not h:raise RuntimeError(lib.dm_error().decode())
 handles.append(h);return h
try:
 captured=torch.load('/tail/captured.pt',weights_only=False)['attention'];q,k,v,mask,scale=captured[0]['args'];b,heads,t,d=q.shape;n=(t+15)//16*16;kk=n
 qk=create(t,d,n);pv=create(t,kk,d)
 with torch.inference_mode():
  for rep in range(3):
   for layer,item in enumerate(captured):
    q,k,v,mask,scale=item['args'];ys=[];marks={};start=time.perf_counter();q=q.float();k=k.float();v=v.float()
    for h in range(heads):
     kh=h//(heads//k.shape[1]);bb=torch.zeros((d,n),dtype=torch.float16);bb[:,:t]=k[0,kh].t().half();a=q[0,h].half().contiguous();scores=torch.empty((t,n),dtype=torch.float32)
     tt=time.perf_counter();check(lib.dm_set_b(qk,bb.data_ptr()));check(lib.dm_run(qk,a.data_ptr(),scores.data_ptr(),t));marks['qk']=marks.get('qk',0)+time.perf_counter()-tt
     scores=scores[:,:t]*scale;scores[:,~mask.reshape(-1)]=-float('inf');prob=torch.softmax(scores,dim=-1);aa=torch.zeros((t,kk),dtype=torch.float16);aa[:,:t]=prob.half();vb=torch.zeros((kk,d),dtype=torch.float16);vb[:t]=v[0,kh].half();out=torch.empty((t,d),dtype=torch.float32)
     tt=time.perf_counter();check(lib.dm_set_b(pv,vb.data_ptr()));check(lib.dm_run(pv,aa.data_ptr(),out.data_ptr(),t));marks['pv']=marks.get('pv',0)+time.perf_counter()-tt;ys.append(out.bfloat16())
    out=torch.stack(ys,dim=1).unsqueeze(0);dt=time.perf_counter()-start;ref=item['reference'];delta=(out.float()-ref.float()).abs();rec={'rep':rep,'layer':layer,'seconds':dt,'max_delta':float(delta.max()),'mean_delta':float(delta.mean()),'allclose':torch.allclose(out.float(),ref.float(),atol=.002,rtol=.008),'native':marks};r['records'].append(rec);print('NPU',rec,flush=True)
 r['completed']=True
except BaseException as exc:
 r['error']=repr(exc);raise
finally:
 for h in reversed(handles):lib.dm_destroy(h)
 r['handles_destroyed']=len(handles);(p/'logs/probe-npu-attention.json').write_text(json.dumps(r,indent=2)+'\n')
