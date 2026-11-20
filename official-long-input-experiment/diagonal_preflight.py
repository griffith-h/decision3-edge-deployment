"""CPU-only pinned sparse reduction comparison; no model or labels."""
import json,time
import torch
from sparse_cpu_backend import BlasRepair,DiagonalRepair

def run():
    torch.set_num_threads(3);torch.manual_seed(43);records=[]
    with torch.inference_mode():
        for k,n in ((64,256),(1024,2048),(2048,1024),(6144,1024),(10240,128)):
            layer=torch.nn.Linear(k,n,bias=False,dtype=torch.bfloat16);x=torch.randn(129,k,dtype=torch.bfloat16)
            mask=torch.rand(129,n)<.15;pairs=mask.nonzero();original=torch.full((129,n),77.,dtype=torch.bfloat16)
            start=time.perf_counter();BlasRepair().run(layer,x,pairs,original);seconds=time.perf_counter()-start
            variants=[]
            for block in (352,1024):
                out=torch.full_like(original,77.);start=time.perf_counter();DiagonalRepair(block).run(layer,x,pairs,out);duration=time.perf_counter()-start
                variants.append({'block':block,'seconds':duration,'mismatches_vs_previous_repair':int((out!=original).sum()),'unselected_untouched':bool((out[~mask]==77).all())})
            records.append({'shape':[129,k,n],'selected':len(pairs),'previous_seconds':seconds,'variants':variants})
    return {'scope':'Synthetic CPU only, sparse arithmetic vs previous public4x4, not actual model acceptance','records':records}

if __name__=='__main__':print(json.dumps(run()))
