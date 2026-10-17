import socket,struct
from pathlib import Path
p=Path('/opt/decision3-edge-cpu-tail-20261017/gpu-worker.sock')
if p.exists():
 s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);s.settimeout(10);s.connect(str(p));s.sendall(struct.pack('!6I',0,0,0,0,0,0));assert s.recv(4)==b'\0\0\0\0';s.close()
