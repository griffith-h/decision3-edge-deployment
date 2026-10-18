import ctypes,numpy as np
from pathlib import Path
lib=ctypes.CDLL(str(Path(__file__).with_name('librow_graph_runtime.so')))
lib.row_graph_error.restype=ctypes.c_char_p;lib.row_graph_create.argtypes=[ctypes.c_char_p];lib.row_graph_create.restype=ctypes.c_void_p
lib.row_graph_input_name.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.row_graph_input_name.restype=ctypes.c_char_p
lib.row_graph_run.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int),ctypes.c_void_p];lib.row_graph_run.restype=ctypes.c_int
lib.row_graph_destroy.argtypes=[ctypes.c_void_p];lib.row_graph_destroy.restype=None
lib.row_graph_rows.argtypes=[ctypes.c_void_p];lib.row_graph_rows.restype=ctypes.c_int
class Graph:
 def __init__(self,path):
  self.handle=lib.row_graph_create(str(path).encode())
  if not self.handle:raise RuntimeError(lib.row_graph_error().decode())
  self.rows=lib.row_graph_rows(self.handle)
  name=lib.row_graph_input_name(self.handle,0)
  if name!=b'x':self.close();raise ValueError('Unexpected MLP input name '+str(name))
 def run(self,x):
  x=np.ascontiguousarray(x,dtype=np.float32);assert x.shape==(1,self.rows,1024);out=np.empty_like(x)
  ptrs=(ctypes.c_void_p*1)(x.ctypes.data);sizes=(ctypes.c_int*1)(x.size)
  if lib.row_graph_run(self.handle,ptrs,sizes,out.ctypes.data):raise RuntimeError(lib.row_graph_error().decode())
  return out
 def close(self):
  if self.handle:lib.row_graph_destroy(self.handle);self.handle=None
lib.row_graph_run_bf16.argtypes=[ctypes.c_void_p]*3;lib.row_graph_run_bf16.restype=ctypes.c_int
lib.row_graph_meta.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)];lib.row_graph_meta.restype=ctypes.c_int
class HalfGraph(Graph):
 def run_bf16(self,x):
  import torch
  if x.dtype!=torch.bfloat16 or x.device.type!='cpu' or x.shape!=(1,self.rows,1024) or torch.is_grad_enabled():raise ValueError('Unsupported half graph signature')
  x=x.contiguous();out=torch.empty_like(x)
  status=lib.row_graph_run_bf16(self.handle,x.data_ptr(),out.data_ptr())
  if status in (-2,-3):raise ValueError('MLP graph input/output outside FP16 safe range')
  if status:raise RuntimeError(lib.row_graph_error().decode())
  return out

lib.row_graph_set_core.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.row_graph_set_core.restype=ctypes.c_int
