"""CPU-only exact comparison to r031's public4x4 repair, no model or labels."""
import json,time
import torch
from sparse_cpu_backend import BlasRepair,RowBlasRepair

def run():
    torch.set_num_threads(3);torch.manual_seed(41);records=[]
    with torch.inference_mode():
        for k,n in ((1024,2048),(6144,1024)):
            layer=torch.nn.Linear(k,n,bias=False,dtype=torch.bfloat16);x=torch.randn(257,k,dtype=torch.bfloat16)
            mask=torch.rand(257,n)<.04;pairs=mask.nonzero();original=torch.full((257,n),77.,dtype=torch.bfloat16)
            start=time.perf_counter();BlasRepair().run(layer,x,pairs,original);seconds=time.perf_counter()-start
            variants=[]
            for mode in (2,):
                out=torch.full_like(original,77.);start=time.perf_counter();RowBlasRepair(mode).run(layer,x,pairs,out);duration=time.perf_counter()-start
                variants.append({'mode':mode,'seconds':duration,'mismatches_vs_previous_repair':int((out!=original).sum()),'unselected_untouched':bool((out[~mask]==77).all())})
            records.append({'shape':[257,k,n],'selected':len(pairs),'previous_seconds':seconds,'variants':variants})
    return {'scope':'Synthetic CPU only; fixed same positions vs prior4x4 repair, not actual model acceptance','records':records}

if __name__=='__main__':print(json.dumps(run()))
