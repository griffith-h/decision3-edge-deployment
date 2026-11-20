import json,time
import torch
from sparse_cpu_backend import Preparation

def run():
    torch.set_num_threads(3);torch.manual_seed(42);records=[]
    raw=torch.randn(512,6144);xn=torch.rand(512)+10;xl=torch.rand(512)*1e-8;hn=xn.clone();wn=torch.rand(6144)+1;wl=torch.rand(6144)*1e-8
    for vector in (False,True):
        state=Preparation(vector);start=time.perf_counter();got=state.mask(raw,xn,xl,hn,wn,wl,1e-7,.1,65536);seconds=time.perf_counter()-start
        if not vector:expected=got
        records.append({'vector':vector,'seconds':seconds,'positions':len(got),'exact_same_positions':bool(torch.equal(got,expected))})
    return {'scope':'CPU-only same operation/order/positions preflight, no model or NPU','records':records}
if __name__=='__main__':print(json.dumps(run()))
