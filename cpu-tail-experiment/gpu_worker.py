"""Private host GPU worker, Unix socket + process-owned shared file. No TCP."""
import ctypes as C,json,mmap,os,socket,struct,time,signal
from pathlib import Path
root=Path('/opt/decision3-edge-cpu-tail-20261017');P=C.c_void_p;I=C.c_int;REGION=8*1024*1024
lib=C.CDLL(str(root/'libgpu_host.so'));lib.d3_gpu_create.argtypes=[C.c_char_p];lib.d3_gpu_create.restype=P;lib.d3_gpu_error.restype=C.c_char_p;lib.d3_gpu_destroy.argtypes=[P]
lib.d3_gpu_attention.argtypes=[P]*6+[I]*4+[C.c_float];lib.d3_gpu_attention.restype=I
lib.d3_gpu_scan.argtypes=[P]*7+[I]*4;lib.d3_gpu_scan.restype=I
path=root/'gpu-worker.sock';bufpath=root/'gpu-ipc.bin';h=lib.d3_gpu_create(str(root/'gpu_kernels.cl').encode())
if not h:raise RuntimeError(lib.d3_gpu_error().decode())
# Launcher guarantees no worker owns these files when preparing this process.
if path.exists():raise RuntimeError('Existing GPU socket; check prior worker ownership')
fd=os.open(bufpath,os.O_RDWR|os.O_CREAT|os.O_TRUNC,0o600);os.ftruncate(fd,6*REGION);mem=mmap.mmap(fd,6*REGION);os.close(fd)
base=C.addressof(C.c_char.from_buffer(mem));ptr=[base+i*REGION for i in range(6)];server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(str(path));os.chmod(path,0o600);server.listen(1);records=[];stop=False
print('GPU READY',os.getpid(),{k:os.environ.get(k) for k in ('D3_GPU_BLAST','D3_GPU_FAST_SCAN','RUSTICL_ENABLE')},flush=True)
def receive(conn,n):
 out=b''
 while len(out)<n:
  data=conn.recv(n-len(out))
  if not data:return None
  out+=data
 return out
try:
 while not stop:
  conn,_=server.accept()
  with conn:
   while True:
    raw=receive(conn,24)
    if raw is None:break
    op,t,heads,x,d,seq=struct.unpack('!6I',raw)
    if op==0:stop=True;conn.sendall(struct.pack('!i',0));break
    started=time.perf_counter()
    if op==1:rc=lib.d3_gpu_attention(h,ptr[0],ptr[1],ptr[2],ptr[3],ptr[5],t,heads,x,d,d**-.5)
    elif op==2:rc=lib.d3_gpu_scan(h,ptr[0],ptr[1],ptr[2],ptr[3],ptr[4],ptr[5],t,heads,x,d)
    else:rc=-1
    records.append({'op':op,'length':t,'heads':heads,'x':x,'d':d,'seq':seq,'seconds':time.perf_counter()-started,'rc':rc})
    conn.sendall(struct.pack('!i',rc))
    if rc:
     error=lib.d3_gpu_error() or b'unsupported operation';conn.sendall(struct.pack('!I',len(error))+error)
finally:
 server.close();path.unlink(missing_ok=True);lib.d3_gpu_destroy(h);del ptr;mem.close()
 (root/'logs/gpu-worker-records.json').write_text(json.dumps(records,indent=2)+'\n')
 print('GPU STOPPED',len(records),flush=True)
