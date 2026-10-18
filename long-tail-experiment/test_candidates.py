import sys
sys.path.insert(0,'/long')
import baseline
import torch,json,threading,time
from pathlib import Path
import long_kernels as lk
import tail_native as n
from fast_scan import TunedKernels
from mlp_schedule import Scheduler
P=Path('/experiment');selected=json.loads((P/'selected.json').read_text())
originals=(lk.attention,lk.scan_prepared,lk.Kernels)
class TestKernels(TunedKernels):
    def __init__(self):super().__init__();self.fast_validate=True;self.scan_config=selected['scan']
lk.attention=lambda *args:n.attention(*args,**selected['attention'])
lk.scan_prepared=lambda *args:n.scan(*args,**selected['scan'])
lk.Kernels=TestKernels
try:
    # Reuse the exact previous boundary/semantic suite; route its failure injection
    # to the new scan backend rather than the old adapter's private function.
    source=Path('/long/test_kernels.py').read_text().replace('oldfn=lk.scan_prepared','oldfn=_tail_backend.scan').replace('lk.scan_prepared=fail','_tail_backend.scan=fail').replace('lk.scan_prepared=oldfn','_tail_backend.scan=oldfn').replace('def fail(*a):','def fail(*a,**kw):')
    source='import tail_native as _tail_backend\n'+source
    (P/'logs/adapted-native-unit.py').write_text(source)
    exec(compile(source,'adapted-native-unit.py','exec'),{})
finally:lk.attention,lk.scan_prepared,lk.Kernels=originals

torch.set_num_threads(3);torch.manual_seed(622)
result={'finite':[],'raw_domains':[],'scheduler':{},'passed':False}
with torch.inference_mode():
    for dtype in (torch.float32,torch.bfloat16):
        for length in (1,511,512,513,2048,4097):
            x=torch.randn(length,dtype=dtype)
            assert n.finite(x)==bool(torch.isfinite(x).all())
            for bad in (float('nan'),float('inf'),-float('inf')):
                y=x.clone();y[-1]=bad;assert not n.finite(y)
            result['finite'].append([str(dtype),length])
    for gd in (torch.float32,torch.bfloat16):
        for bd in (torch.float32,torch.bfloat16):
            q=torch.randn(2,2,513,4).bfloat16().transpose(1,2);k=torch.randn_like(q);v=torch.randn(2,2,513,8).bfloat16().transpose(1,2)
            g=-torch.rand(2,2,513).to(gd).transpose(1,2);b=torch.rand(2,2,513).to(bd).transpose(1,2)
            xs=(q,k,v,g,b);assert n.validate_raw(*xs)
            for i,bad in [(0,float('nan')),(1,float('inf')),(2,-float('inf')),(3,.01),(3,float('nan')),(4,-.01),(4,1.1),(4,float('inf'))]:
                copied=[x.clone() for x in xs];copied[i][0,512,0]=bad;assert not n.validate_raw(*copied),(gd,bd,i,bad)
            g.zero_();b.zero_();assert n.validate_raw(q,k,v,g,b);b.fill_(1);assert n.validate_raw(q,k,v,g,b)
            result['raw_domains'].append([str(gd),str(bd)])

    done=threading.Event();fault={'enabled':False};bank_calls=[]
    class Graph:
        def __init__(self,width):self.width=width;self.calls=0;self.busy=False
        def run_bf16(self,x):
            assert not self.busy;self.busy=True
            try:
                self.calls+=1
                if self.width==384:time.sleep(.03);done.set()
                if fault['enabled'] and self.width==128:
                    if self.calls==2:raise RuntimeError('injected second graph call')
                    return x*11
                return x*3
            finally:self.busy=False
    class Bank:
        def run_rows(self,g,x):
            if fault['enabled']:assert done.is_set(),'fallback before other worker finished'
            bank_calls.append(tuple(x.shape));return x*3
    g=Graph(115);a=Graph(128);z=Graph(384);bank=Bank();bank.extra=[a,z];bank.by_id={id(g):{128:a,384:z}};bank.plans={615:(128,128,384)};bank.counts={'graph_calls':0,'padded_rows':0};masks={id(a):0,id(z):0}
    def set_core(h,mask):masks[id(h)]=mask;return 0
    sched=Scheduler(bank,set_core);sched.enabled=True
    x=(torch.arange(615).float()[:,None]%97).expand(615,1024).contiguous().bfloat16().unsqueeze(0)
    try:
        assert torch.equal(sched.run(g,x),x*3) and sched.counts['parallel_layers']==1
        assert masks[id(a)]==2 and masks[id(z)]==1 and a.calls==2 and z.calls==1
        fault['enabled']=True;a.calls=z.calls=0;done.clear()
        assert torch.equal(sched.run(g,x),x*3) and id(g) in sched.disabled and len(bank_calls)==1
        assert all(mask==0 for mask in masks.values()) and not a.busy and not z.busy
        assert torch.equal(sched.run(g,x),x*3) and len(bank_calls)==2
        fault['enabled']=False;assert torch.equal(sched.run(g,x[:,:512]),x[:,:512]*3)
        result['scheduler']={'passed':True,'counts':dict(sched.counts),'joined_before_fallback':True,'row_order_preserved':True,'fault_group_disabled':True}
    finally:sched.close();sched.close()
result['passed']=True;(P/'logs/extra-unit.json').write_text(json.dumps(result,indent=2)+'\n');print('EXTRA UNIT PASS',result,flush=True)
