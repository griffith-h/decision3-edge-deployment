import ctypes,json,torch
from pathlib import Path
p=Path('/experiment');lib=ctypes.CDLL(str(p/'libnative_probe.so'));lib.mm_create_core.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int];lib.mm_create_core.restype=ctypes.c_void_p;lib.mm_destroy.argtypes=[ctypes.c_void_p];lib.mm_set_output_precision(1);lib.mm_native_attrs.argtypes=[ctypes.c_void_p,ctypes.c_void_p];lib.mm_native_attrs.restype=ctypes.c_int;lib.mm_error.restype=ctypes.c_char_p
r=[]
for k,n in ((128,256),(1024,3072),(3072,1024)):
 b=torch.zeros(k,n,dtype=torch.float16);h=lib.mm_create_core(k,n,b.data_ptr(),1);assert h,lib.mm_error();out=(ctypes.c_int*(4*3*12))()
 try:
  count=lib.mm_native_attrs(h,out);r.append({'k':k,'n':n,'attrs':[[list(out[(i*3+j)*12:(i*3+j+1)*12]) for j in range(3)] for i in range(count)]})
 finally:lib.mm_destroy(h)
(p/'logs/native-fp32-attrs.json').write_text(json.dumps(r,indent=2));print('NATIVE ATTRS',r[0],flush=True)
