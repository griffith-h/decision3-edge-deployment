import ctypes,json
from pathlib import Path
C=ctypes;L=C.CDLL('libOpenCL.so.1');P=C.c_void_p;I=C.c_uint
L.clGetPlatformIDs.argtypes=[I,C.POINTER(P),C.POINTER(I)];L.clGetPlatformIDs.restype=C.c_int
L.clGetDeviceIDs.argtypes=[P,C.c_ulong,I,C.POINTER(P),C.POINTER(I)];L.clGetDeviceIDs.restype=C.c_int
L.clGetDeviceInfo.argtypes=[P,I,C.c_size_t,P,C.POINTER(C.c_size_t)];L.clGetDeviceInfo.restype=C.c_int
n=I();code=L.clGetPlatformIDs(0,None,C.byref(n));r={'platform_code':code,'platform_count':n.value,'devices':[]}
if code==0:
 platforms=(P*n.value)();assert L.clGetPlatformIDs(n,platforms,None)==0
 for p in platforms:
  count=I();code=L.clGetDeviceIDs(p,4,0,None,C.byref(count));r.setdefault('device_queries',[]).append({'code':code,'count':count.value})
  if code:continue
  devices=(P*count.value)();assert L.clGetDeviceIDs(p,4,count,devices,None)==0
  for d in devices:
   row={}
   for key,param in [('name',0x102b),('version',0x102f),('extensions',0x1030)]:
    size=C.c_size_t();assert L.clGetDeviceInfo(d,param,0,None,C.byref(size))==0;buf=C.create_string_buffer(size.value);assert L.clGetDeviceInfo(d,param,size,buf,None)==0;row[key]=buf.value.decode()
   r['devices'].append(row)
Path('/opt/decision3-edge-cpu-tail-20261017/logs/gpu-private-runtime.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2),flush=True)
