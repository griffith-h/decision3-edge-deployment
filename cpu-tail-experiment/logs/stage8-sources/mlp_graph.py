import ctypes,numpy as np
from pathlib import Path
lib=ctypes.CDLL(str(Path(__file__).with_name('libmlp_graph_runtime.so')))
lib.mlp_graph_error.restype=ctypes.c_char_p;lib.mlp_graph_create.argtypes=[ctypes.c_char_p];lib.mlp_graph_create.restype=ctypes.c_void_p
lib.mlp_graph_input_name.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.mlp_graph_input_name.restype=ctypes.c_char_p
lib.mlp_graph_run.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int),ctypes.c_void_p];lib.mlp_graph_run.restype=ctypes.c_int
lib.mlp_graph_destroy.argtypes=[ctypes.c_void_p];lib.mlp_graph_destroy.restype=None
class Graph:
 def __init__(self,path):
  self.handle=lib.mlp_graph_create(str(path).encode())
  if not self.handle:raise RuntimeError(lib.mlp_graph_error().decode())
  name=lib.mlp_graph_input_name(self.handle,0)
  if name!=b'x':self.close();raise ValueError('Unexpected MLP input name '+str(name))
 def run(self,x):
  x=np.ascontiguousarray(x,dtype=np.float32);assert x.shape==(1,115,1024);out=np.empty_like(x)
  ptrs=(ctypes.c_void_p*1)(x.ctypes.data);sizes=(ctypes.c_int*1)(x.size)
  if lib.mlp_graph_run(self.handle,ptrs,sizes,out.ctypes.data):raise RuntimeError(lib.mlp_graph_error().decode())
  return out
 def close(self):
  if self.handle:lib.mlp_graph_destroy(self.handle);self.handle=None
