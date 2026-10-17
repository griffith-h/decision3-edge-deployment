"""Check actual Inductor kernels, BF16 boundaries and noncontiguous layouts."""
import json,time
from pathlib import Path
import torch
import torch._inductor.config as config
from compiled_regions import mlp_region,norm_region,gated_region,install
from types import SimpleNamespace
torch.set_num_threads(3);config.emulate_precision_casts=True;config.cpp.threads=3;config.compile_threads=1
records=[];torch.manual_seed(19)
with torch.inference_mode():
    for name,fn in [('mlp',mlp_region),('norm',norm_region),('gated',gated_region)]:
        compiled=torch.compile(fn,fullgraph=True,dynamic=True,backend='inductor')
        for rows in (1,17,115,129):
            x=(torch.randn(rows,256)*2).bfloat16()[:,::2];w=torch.randn(128).bfloat16();g=torch.randn_like(x)
            args=(x,g) if name=='mlp' else (x,w,1e-6) if name=='norm' else (x,w,g,1e-6)
            t=time.perf_counter();actual=compiled(*args);elapsed=time.perf_counter()-t;expected=fn(*args)
            assert actual.shape==expected.shape and actual.dtype==expected.dtype
            assert torch.isfinite(actual).all()
            delta=(actual.float()-expected.float()).abs().max().item()
            # One BF16 ulp for floating point reduction/vector exp differences.
            tolerance=(expected.float().abs()/128+.0001)
            assert ((actual.float()-expected.float()).abs()<=tolerance).all(),(name,rows,delta)
            records.append({'region':name,'rows':rows,'max_abs_delta':delta,'first_call_seconds':elapsed,'exact':torch.equal(actual,expected)})
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
backbone=torch.nn.Module();backbone.language_model=torch.nn.Module()
mlp=qwen.Qwen3_5MLP(SimpleNamespace(hidden_size=64,hidden_act='silu'),128).bfloat16()
norm=qwen.Qwen3_5RMSNorm(64).bfloat16();gated=qwen.Qwen3_5RMSNormGated(64).bfloat16()
backbone.language_model.layers=torch.nn.ModuleList([mlp,norm,gated]);toy=SimpleNamespace(backbone=backbone)
originals=[m.forward for m in (mlp,norm,gated)];control={'mode':'all'};metadata={};restore=install(toy,control,metadata)
try:
    with torch.inference_mode():
        for m,old in zip((mlp,norm,gated),originals):
            for rows in (0,3):
                x=torch.randn(rows,64).bfloat16();args=(x,torch.randn_like(x)) if m is gated else (x,)
                torch.testing.assert_close(m(*args),old(*args),rtol=.01,atol=.0001)
        x=torch.randn(3,64);torch.testing.assert_close(norm(x),originals[1](x),rtol=0,atol=0)
        control['mode']='none';x=torch.randn(3,64).bfloat16()
        assert torch.equal(mlp(x),originals[0](x));control['mode']='all'
    before=dict(metadata['calls']);x=torch.randn(3,64,dtype=torch.bfloat16,requires_grad=True)
    mlp(x).float().sum().backward();assert x.grad is not None and metadata['calls']==before
    assert not metadata['errors'] and all(metadata['calls'].get(n,0)>0 for n in ('mlp','norm','gated'))
finally:restore()
assert all(m.forward==old for m,old in zip((mlp,norm,gated),originals))
result={'passed':True,'precision_cast_emulation':True,'records':records,'empty_float32_grad_disabled_and_restore_checks':True,'wrapper_stats':metadata}
Path('/work/logs/compile-unit.json').write_text(json.dumps(result,indent=2)+'\n')
print('COMPILE_UNITS_PASSED',flush=True)
