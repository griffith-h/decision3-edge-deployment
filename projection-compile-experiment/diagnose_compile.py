"""Diagnostic, not an acceptance test: localize Inductor rounding differences."""
import json,time
from pathlib import Path
import torch
import torch._inductor.config as config
from compiled_regions import mlp_region,norm_region,gated_region
torch.set_num_threads(3);config.emulate_precision_casts=True;config.cpp.threads=3;config.compile_threads=1
torch.manual_seed(19);records=[]
def ordered(x):
    b=x.contiguous().view(torch.uint16).int()
    return torch.where(b<32768,b+32768,65535-b)
with torch.inference_mode():
    for name,fn in [('mlp',mlp_region),('norm',norm_region),('gated',gated_region)]:
        compiled=torch.compile(fn,fullgraph=True,dynamic=True,backend='inductor')
        for rows in (1,17,115,129):
            x=(torch.randn(rows,256)*2).bfloat16()[:,::2];w=torch.randn(128).bfloat16();g=torch.randn_like(x)
            args=(x,g) if name=='mlp' else (x,w,1e-6) if name=='norm' else (x,w,g,1e-6)
            actual=compiled(*args);expected=fn(*args)
            delta=(actual.float()-expected.float()).abs();ulp=(ordered(actual)-ordered(expected)).abs()
            idx=delta.argmax().item();a=actual.flatten()[idx].item();e=expected.flatten()[idx].item()
            r={'region':name,'rows':rows,'max_abs_delta':delta.max().item(),'max_bf16_ulp_distance':ulp.max().item(),'different_elements':int((actual!=expected).sum()),'total_elements':actual.numel(),'example_max_abs':{'actual':a,'expected':e,'ulp_distance':ulp.flatten()[idx].item()}}
            assert torch.isfinite(actual).all();records.append(r);print(r,flush=True)
Path('/work/logs/compile-diagnostic.json').write_text(json.dumps({'diagnostic_only':True,'records':records},indent=2)+'\n')
