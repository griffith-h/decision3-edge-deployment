import ctypes as C,json
from pathlib import Path
root=Path('/opt/decision3-edge-cpu-tail-20261017');P=C.c_void_p;I=C.c_int;lib=C.CDLL(str(root/'libgpu_host.so'));lib.d3_gpu_create.argtypes=[C.c_char_p];lib.d3_gpu_create.restype=P;lib.d3_gpu_error.restype=C.c_char_p;lib.d3_gpu_destroy.argtypes=[P]
lib.d3_gpu_attention.argtypes=[P]*6+[I]*4+[C.c_float];lib.d3_gpu_attention.restype=I
h=lib.d3_gpu_create(str(root/'gpu_kernels.cl').encode());assert h,lib.d3_gpu_error()
try:
 for t in (1,3,115):
  heads,kh,d=8,2,32;q=(C.c_uint16*(heads*t*d))(*([0x3e00]*(heads*t*d)));k=(C.c_uint16*(kh*t*d))(*([0x3e80]*(kh*t*d)));v=(C.c_uint16*(kh*t*d))(*([0x3f80]*(kh*t*d)));mask=(C.c_uint8*t)(*([1]*t));out=(C.c_uint16*(heads*t*d))()
  rc=lib.d3_gpu_attention(h,q,k,v,mask,out,t,heads,kh,d,d**-.5);print('OWN DEBUG',t,rc,lib.d3_gpu_error(),list(out)[:10],list(out)[-10:],flush=True)
finally:lib.d3_gpu_destroy(h)
