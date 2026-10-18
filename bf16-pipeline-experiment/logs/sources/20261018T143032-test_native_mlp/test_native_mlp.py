"""Actual weights/615 activations; native vs same FP32 normal-layout GEMMs."""
import sys
sys.path.insert(0,'/tail')
from runner import *
from native_mlp import NativeMlp
from precise_mlp import PreciseMlp
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from pipeline import configure
from lut_ops import Tables
import ctypes
p=Path('/experiment');torch.set_num_threads(3);lk.blas.scipy_openblas_set_num_threads64_(1);lib=ctypes.CDLL('/fusion/libmixed_bridge.so');lib.mm_create_core.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int];lib.mm_create_core.restype=ctypes.c_void_p;lib.mm_destroy.argtypes=[ctypes.c_void_p];lib.mm_error.restype=ctypes.c_char_p;lib.mm_set_output_precision(1)
for name,types in [('mm_prepare_bf16',[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]),('mm_copy_prepared',[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]),('mm_run_raw',[ctypes.c_void_p,ctypes.c_int]),('mm_run_prepared_bf16',[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p])]:getattr(lib,name).argtypes=types;getattr(lib,name).restype=ctypes.c_int
handles={}
def status(rc):
 if rc in (-2,-3):raise ValueError(rc)
 if rc:raise RuntimeError(repr(lib.mm_error()))
def ensure(m):
 if id(m) not in handles:
  w=m.weight.detach().half().t().contiguous();h=lib.mm_create_core(m.in_features,m.out_features,w.data_ptr(),engine.core.get(id(m),0));assert h,lib.mm_error();handles[id(m)]=h
 return handles[id(m)]
engine=SimpleNamespace(lib=lib,pool=ThreadPoolExecutor(2),core={},calls=0,status=status,ensure=ensure);table=Tables();native=NativeMlp(engine,table);normal=PreciseMlp(SimpleNamespace(backbone=torch.nn.Identity()),engine,table,width=640);weights=torch.load('/work/mlp-layers.pt',weights_only=False);capture=torch.load('/tail/captured.pt',weights_only=False)['mlp'];r={'records':[],'completed':False};banks=[]
try:
 with torch.inference_mode():
  for i in (0,5,11):
   members=[]
   for name in ('gate_proj','up_proj','down_proj'):
    w=weights[i]['weights'][name];m=torch.nn.Linear(w.shape[1],w.shape[0],bias=False,dtype=torch.bfloat16);m.weight=torch.nn.Parameter(w,requires_grad=False);members.append(m)
   banks.append(members);engine.core.update({id(members[0]):1,id(members[1]):2,id(members[2]):1});x=capture[i]['input'];ref=normal.run(members,x);out=native.run(members,x);delta=float((out.float()-ref.float()).abs().max());print('NATIVE EQ',i,delta,flush=True);assert torch.equal(out,ref),'Native C/A layout or arithmetic differs'
   for _ in range(3):normal.run(members,x);native.run(members,x)
   for rep in range(12):
    for mode in (('normal','native') if rep%2==0 else ('native','normal')):
     t=time.perf_counter();out=(normal if mode=='normal' else native).run(members,x);dt=time.perf_counter()-t;assert torch.equal(out,ref);r['records'].append({'layer':i,'pair':rep,'mode':mode,'seconds':dt,'bit_equal':True})
   print('NATIVE PERF',i,{m:stats([x['seconds'] for x in r['records'] if x['layer']==i and x['mode']==m]) for m in ('normal','native')},flush=True)
 r['completed']=True
finally:
 engine.pool.shutdown(wait=True);native.close();native.close();normal.close()
 for h in handles.values():lib.mm_destroy(h)
 handles.clear();r['normal_handles_after_close']=len(handles);r['native_handles_after_close']=len(native.handles);(p/'logs/native-mlp-unit.json').write_text(json.dumps(r,indent=2))
