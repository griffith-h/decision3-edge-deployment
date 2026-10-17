import ctypes,json,time
from pathlib import Path
import numpy as np
lib=ctypes.CDLL(str(Path(__file__).with_name('libgraph_runtime.so')))
lib.graph_error.restype=ctypes.c_char_p
lib.graph_create.argtypes=[ctypes.c_char_p];lib.graph_create.restype=ctypes.c_void_p
lib.graph_input_name.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.graph_input_name.restype=ctypes.c_char_p
lib.graph_run.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int),ctypes.c_void_p];lib.graph_run.restype=ctypes.c_int
lib.graph_destroy.argtypes=[ctypes.c_void_p]
class Graph:
    def __init__(self,path):
        self.handle=lib.graph_create(str(path).encode())
        if not self.handle:raise RuntimeError(lib.graph_error().decode())
        self.names=[lib.graph_input_name(self.handle,i).decode() for i in range(4)]
        if set(self.names)!={'x','cos','sin','mask'}:self.close();raise ValueError(self.names)
    def run(self,inputs):
        values=[np.ascontiguousarray(inputs[n],dtype=np.float32) for n in self.names]
        ptrs=(ctypes.c_void_p*4)(*[x.ctypes.data for x in values]);sizes=(ctypes.c_int*4)(*[x.size for x in values]);out=np.empty((1,115,1024),dtype=np.float32)
        rc=lib.graph_run(self.handle,ptrs,sizes,out.ctypes.data)
        if rc:raise RuntimeError(lib.graph_error().decode())
        return out
    def close(self):
        if self.handle:lib.graph_destroy(self.handle);self.handle=None
