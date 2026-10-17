import ctypes,torch,json,time
from pathlib import Path
from baseline import stats
r={};p=ctypes.c_void_p
old=ctypes.CDLL('/previous/libconversion_bridge.so');new=ctypes.CDLL('/work/liblut_bridge.so')
for lib in (old,new):
 for name in ('mm_convert_input','mm_fast_convert_output'):
  f=getattr(lib,name);f.argtypes=[p,p,ctypes.c_size_t];f.restype=ctypes.c_int
with torch.inference_mode():
 bits=torch.arange(65536,dtype=torch.int32).short();a=torch.empty_like(bits);b=torch.empty_like(bits)
 for name in ('mm_convert_input','mm_fast_convert_output'):
  f=getattr(old,name);g=getattr(new,name)
  for n in (1,3,4,7,8,9,65535,65536):
   c=f(bits.data_ptr(),a.data_ptr(),n);d=g(bits.data_ptr(),b.data_ptr(),n);assert c==d;assert torch.equal(a[:n],b[:n]),(name,n)
  x=(torch.randn(115,7168)*2).bfloat16()
  if name=='mm_fast_convert_output':x=x.half()
  o=torch.empty_like(x);row={}
  for key,fn in [('old',f),('lut',g)]:
   times=[]
   for _ in range(30):t=time.perf_counter();fn(x.data_ptr(),o.data_ptr(),x.numel());times.append(time.perf_counter()-t)
   row[key]=stats(times)
  r[name]=row
r['passed']=True;Path('/work/logs/bridge-lut-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('BRIDGE LUT PASS',flush=True)
