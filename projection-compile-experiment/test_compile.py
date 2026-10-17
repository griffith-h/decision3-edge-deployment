"""Check actual Inductor kernels, BF16 boundaries and noncontiguous layouts."""
import json,time
from pathlib import Path
import torch
import torch._inductor.config as config
from compiled_regions import mlp_region,norm_region,gated_region,bf16_round_float,install
import torch.nn.functional as F
from types import SimpleNamespace
torch.set_num_threads(3);config.emulate_precision_casts=True;config.cpp.threads=3;config.compile_threads=1
records=[];torch.manual_seed(19)
def eager_gated(x,w,gate,eps):
    z=x.float();z=(z*torch.rsqrt(z.square().mean(-1,keepdim=True)+eps)).to(x.dtype)
    return ((w*z)*F.silu(gate.float())).to(x.dtype)
references={'mlp':lambda x,y:F.silu(x)*y,'norm':norm_region,'gated':eager_gated}
with torch.inference_mode():
    raw=torch.randint(0,2**32,(100003,),dtype=torch.int64).to(torch.uint32).view(torch.float32)
    rounding=torch.compile(bf16_round_float,fullgraph=True,dynamic=True,backend='inductor')
    expected=raw.bfloat16().float();actual=rounding(raw);mask=~torch.isnan(expected)
    assert torch.equal(actual[mask].view(torch.int32),expected[mask].view(torch.int32))
    assert torch.equal(torch.isnan(actual),torch.isnan(expected))
    torch.manual_seed(19)
    for name,fn in [('mlp',mlp_region),('norm',norm_region),('gated',gated_region)]:
        compiled=torch.compile(fn,fullgraph=True,dynamic=True,backend='inductor')
        for rows in (1,17,115,129):
            x=(torch.randn(rows,256)*2).bfloat16()[:,::2];w=torch.randn(128).bfloat16();g=torch.randn_like(x)
            args=(x,g) if name=='mlp' else (x,w,1e-6) if name=='norm' else (x,w,g,1e-6)
            t=time.perf_counter();actual=compiled(*args);elapsed=time.perf_counter()-t;expected=references[name](*args)
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
originals=[m.forward for m in (mlp,norm,gated)];control={'mode':'all'};metadata={};original_rope=qwen.apply_rotary_pos_emb;restore=install(toy,control,metadata)
try:
    with torch.inference_mode():
        for m,old in zip((mlp,norm,gated),originals):
            for rows in (0,3):
                x=torch.randn(rows,64).bfloat16();args=(x,torch.randn_like(x)) if m is gated else (x,)
                torch.testing.assert_close(m(*args),old(*args),rtol=.01,atol=.0001)
        x=torch.randn(3,64);torch.testing.assert_close(norm(x),originals[1](x),rtol=0,atol=0)
        control['mode']='none';x=torch.randn(3,64).bfloat16()
        assert torch.equal(mlp(x),originals[0](x));control['mode']='all'
        q=torch.randn(1,2,17,128).bfloat16();k=torch.randn(1,1,17,128).bfloat16()
        for dtype in (torch.bfloat16,torch.float32,torch.float64):
            c=torch.randn(1,17,64).to(dtype);s=torch.randn_like(c)
            reference=original_rope(q,k,c,s);actual=qwen.apply_rotary_pos_emb(q,k,c,s)
            for z,ref in zip(actual,reference):torch.testing.assert_close(z,ref,rtol=0,atol=0)
        control['freeze_shapes']=True;before_shape=dict(metadata['calls'])
        novel=torch.randn(5,64).bfloat16()
        assert torch.equal(norm(novel),originals[1](novel)) and metadata['calls']==before_shape
        assert metadata['unseen_shape_fallbacks']==1;control['freeze_shapes']=False
    before=dict(metadata['calls']);x=torch.randn(3,64,dtype=torch.bfloat16,requires_grad=True)
    mlp(x).float().sum().backward();assert x.grad is not None and metadata['calls']==before
    assert not metadata['errors'] and all(metadata['calls'].get(n,0)>0 for n in ('mlp','norm','gated','rope'))
finally:restore()
assert all(m.forward==old for m,old in zip((mlp,norm,gated),originals))
result={'passed':True,'precision_cast_emulation':True,'explicit_rounding_random_fp32_values':100003,'records':records,'empty_float32_grad_disabled_and_restore_checks':True,'wrapper_stats':metadata}
Path('/work/logs/compile-unit.json').write_text(json.dumps(result,indent=2)+'\n')
print('COMPILE_UNITS_PASSED',flush=True)
