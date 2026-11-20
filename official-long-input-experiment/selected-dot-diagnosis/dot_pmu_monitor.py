"""Bounded host-only PMU observer; no container capability or sysctl changes.

Attach only to the authenticated Unix peer inside this owned diagnostic worker.
Every thread gets a small non-multiplexed group; never count unrelated tasks.
"""
import argparse,ctypes,fcntl,json,os,socket,struct,subprocess,time
from pathlib import Path

SUITES={
    'core':('cpu_cycles','inst_retired','inst_spec','mem_access','l1d_cache','l1d_cache_refill'),
    'cache':('cpu_cycles','inst_retired','l1d_tlb_refill','l2d_cache','l2d_cache_refill','bus_access'),
    'front':('cpu_cycles','inst_retired','l1i_cache_refill','br_mis_pred','br_pred','inst_spec')}

def decode_group(data,names):
    if len(data)!=8*(3+len(names)):raise ValueError('Incomplete PMU group read')
    values=struct.unpack('Q'*(3+len(names)),data)
    if values[0]!=len(names):raise ValueError('PMU group shape changed')
    return {'enabled_ns':values[1],'running_ns':values[2],
            'counts':dict(zip(names,values[3:])),
            'running_fraction':values[2]/values[1] if values[1] else 1.}

def owned_peer(pid):
    info=json.loads(subprocess.check_output(['docker','inspect','decision3-official-opt'],text=True))[0]
    label=info['Config'].get('Labels',{}).get('decision3.opt.run','')
    if '-selected-dot-diagnosis-' not in label:raise RuntimeError('Peer does not belong to owned diagnostic')
    ancestor=pid;seen=set()
    while ancestor>1 and ancestor not in seen:
        if ancestor==info['State']['Pid']:return label
        seen.add(ancestor);ancestor=int(Path(f'/proc/{ancestor}/stat').read_text().rsplit(')',1)[1].split()[1])
    raise RuntimeError('Peer is outside owned worker process tree')

class Monitor:
    def __init__(self,pid):
        if os.uname().machine!='aarch64':raise RuntimeError('Pinned AArch64 host required')
        self.pid=pid;self.fds=[];self.names=();self.tids=[]
        self.lib=ctypes.CDLL(None,use_errno=True);self.lib.syscall.restype=ctypes.c_long
        root=Path('/sys/bus/event_source/devices/armv8_cortex_a72')
        self.kind=int((root/'type').read_text());self.config={}
        for name in {n for names in SUITES.values() for n in names}:
            text=(root/'events'/name).read_text().strip();self.config[name]=int(text.split('=',1)[1],0)
    def tasks(self):return sorted(int(p.name) for p in Path(f'/proc/{self.pid}/task').iterdir() if p.name.isdigit())
    def close(self):
        for group in self.fds:
            for fd in reversed(group):os.close(fd)
        self.fds=[]
    def begin(self,suite):
        if self.fds:raise RuntimeError('Counters already active')
        self.names=SUITES[suite];self.tids=self.tasks()
        try:
            for tid in self.tids:
                if not os.sched_getaffinity(tid)<=set(range(4,8)):raise RuntimeError('Target thread escaped fixed CPU set')
                group=[];self.fds.append(group)
                for name in self.names:
                    buf=bytearray(128);struct.pack_into('IIQ',buf,0,self.kind,128,self.config[name])
                    struct.pack_into('Q',buf,32,11) # GROUP | TOTAL_TIME_ENABLED | TOTAL_TIME_RUNNING
                    struct.pack_into('Q',buf,40,1|(1<<5)|(1<<6)) # disabled, exclude kernel/hypervisor
                    attr=ctypes.create_string_buffer(bytes(buf))
                    fd=self.lib.syscall(241,ctypes.byref(attr),tid,-1,group[0] if group else -1,0)
                    if fd<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
                    group.append(fd)
            for group in self.fds:fcntl.ioctl(group[0],0x2403,1);fcntl.ioctl(group[0],0x2400,1)
        except BaseException:self.close();raise
    def end(self):
        for group in self.fds:fcntl.ioctl(group[0],0x2401,1)
        try:
            rows=[{'tid':tid,**decode_group(os.read(group[0],8*(3+len(self.names))),self.names)} for tid,group in zip(self.tids,self.fds)]
            changed=self.tasks()!=self.tids
            totals={name:sum(r['counts'][name] for r in rows) for name in self.names}
            active=[r for r in rows if r['counts']['cpu_cycles']>0]
            return {'threads':rows,'totals':totals,'thread_set_changed':changed,
                    'min_running_fraction':min((r['running_fraction'] for r in active),default=1.),
                    'active_threads':len(active),'event_config':{n:self.config[n] for n in self.names}}
        finally:self.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--socket',type=Path,required=True);p.add_argument('--timeout',type=int,default=900);a=p.parse_args()
    os.sched_setaffinity(0,{0,1,2,3})
    a.socket.parent.mkdir(mode=0o700,exist_ok=True);server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    server.settimeout(a.timeout);monitor=None
    try:
        server.bind(str(a.socket));os.chmod(a.socket,0o600);server.listen(1)
        print('PMU_READY '+str(a.socket),flush=True)
        client,_=server.accept();client.settimeout(a.timeout)
        with client:
            pid,uid,gid=struct.unpack('3i',client.getsockopt(socket.SOL_SOCKET,socket.SO_PEERCRED,12))
            if uid!=0:raise RuntimeError('Expected owned root peer')
            label=owned_peer(pid);monitor=Monitor(pid);print('PMU_PEER '+json.dumps({'pid':pid,'label':label}),flush=True)
            stream=client.makefile('rwb',buffering=0)
            while True:
                line=stream.readline(65536)
                if not line:break
                request=json.loads(line);op=request['op']
                if op=='begin':monitor.begin(request['suite']);reply={'ok':True}
                elif op=='end':reply={'ok':True,**monitor.end()}
                elif op=='quit':stream.write(b'{"ok":true}\n');break
                else:raise ValueError('Unexpected PMU operation')
                stream.write((json.dumps(reply)+'\n').encode())
    finally:
        if monitor:monitor.close()
        server.close()
        if a.socket.exists():a.socket.unlink()
        print('PMU_CLOSED',flush=True)

if __name__=='__main__':main()
