import ctypes
NAMES=('mm_error','mm_create_core','mm_run','mm_destroy','mm_prepare_bf16','mm_copy_prepared','mm_run_prepared_bf16','mm_run_bf16','mm_run_prepared_scatter_bf16')
def library(engine,path):
 old=engine.lib;new=ctypes.CDLL(path)
 for name in NAMES:
  src=getattr(old,name);dst=getattr(new,name);dst.argtypes=src.argtypes;dst.restype=src.restype
 return new
