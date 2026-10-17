"""Single-owner synchronous private IPC. Copies are included in model timing."""
import ctypes as C,mmap,socket,struct,torch
from pathlib import Path
REGION=8*1024*1024
class GPU:
 def __init__(self,root=Path('/work')):
  self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.settimeout(30);self.sock.connect(str(root/'gpu-worker.sock'))
  self.file=open(root/'gpu-ipc.bin','r+b');self.mem=mmap.mmap(self.file.fileno(),6*REGION);self.base=C.addressof(C.c_char.from_buffer(self.mem));self.seq=0
 def recv(self,n):
  b=b''
  while len(b)<n:
   data=self.sock.recv(n-len(b))
   if not data:raise RuntimeError('GPU worker disconnected')
   b+=data
  return b
 def run(self,op,inputs,shape,outshape):
  self.seq+=1
  for i,x in enumerate(inputs):
   if x.numel()*x.element_size()>REGION:raise ValueError('GPU input exceeds private buffer')
   x=x.contiguous();C.memmove(self.base+i*REGION,x.data_ptr(),x.numel()*x.element_size())
  self.sock.sendall(struct.pack('!6I',op,*shape,self.seq));status=struct.unpack('!i',self.recv(4))[0]
  if status:
   n=struct.unpack('!I',self.recv(4))[0];raise RuntimeError(self.recv(n).decode())
  out=torch.empty(outshape,dtype=torch.bfloat16);C.memmove(out.data_ptr(),self.base+5*REGION,out.numel()*2);return out
 def attention(self,q,k,v,mask):
  b,h,t,d=q.shape;assert b==1
  return self.run(1,(q,k,v,mask),(t,h,k.shape[1],d),(1,t,h,d))
 def scan(self,q,k,v,decay,beta):
  b,h,t,kd=q.shape;assert b==1
  return self.run(2,(q,k,v,decay,beta),(t,h,kd,v.shape[-1]),(1,t,h,v.shape[-1]))
 def close(self):self.sock.close();self.mem.close();self.file.close()
