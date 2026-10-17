import ctypes,time,json
from pathlib import Path
lib=ctypes.CDLL('/opt/decision3-edge-cpu-tail-20261017/libgpu_host.so');lib.d3_gpu_create.argtypes=[ctypes.c_char_p];lib.d3_gpu_create.restype=ctypes.c_void_p;lib.d3_gpu_error.restype=ctypes.c_char_p;lib.d3_gpu_destroy.argtypes=[ctypes.c_void_p]
t=time.perf_counter();h=lib.d3_gpu_create(b'/opt/decision3-edge-cpu-tail-20261017/gpu_kernels.cl');r={'created':bool(h),'seconds':time.perf_counter()-t,'error':None if h else lib.d3_gpu_error().decode()}
if h:lib.d3_gpu_destroy(h)
Path('/opt/decision3-edge-cpu-tail-20261017/logs/gpu-kernel-build.json').write_text(json.dumps(r,indent=2)+'\n');print(r,flush=True)
