import ctypes,json
from pathlib import Path
p=ctypes.c_void_p;lib=ctypes.CDLL('/work/libnative_probe.so');lib.mm_create_core.argtypes=[ctypes.c_int,ctypes.c_int,p,ctypes.c_int];lib.mm_create_core.restype=p;lib.mm_destroy.argtypes=[p];lib.mm_native_attrs.argtypes=[p,ctypes.POINTER(ctypes.c_int)];lib.mm_error.restype=ctypes.c_char_p
w=(ctypes.c_uint16*(32*32))();h=lib.mm_create_core(32,32,w,1)
if not h:raise RuntimeError(lib.mm_error())
try:
 a=(ctypes.c_int*36)();assert lib.mm_native_attrs(h,a)==0;r={name:list(a[i*12:i*12+12]) for i,name in enumerate(('A','B','C'))};Path('/work/logs/native-layout-attrs.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
finally:lib.mm_destroy(h)
