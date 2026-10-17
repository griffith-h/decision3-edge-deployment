"""Focused native equation, causality, ownership and fallback checks."""
import json
from pathlib import Path
import torch
from scan import make_scan
from cpu_attention_optimized import chunk_delta_prefill
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

def run():
    torch.manual_seed(3576)
    native=make_scan(lambda *a,**k:chunk_delta_prefill(*a,**k,chunk_size=128))
    checks=[]
    def inputs(b,t,h,k,v):
        # Noncontiguous slices exercise the owned preparation buffers.
        q=torch.randn(b,t,h,k*2,dtype=torch.bfloat16)[...,::2]
        key=torch.randn_like(q)
        value=torch.randn(b,t,h,v,dtype=torch.bfloat16)
        g=-torch.rand(b,t,h)*.3
        beta=torch.rand(b,t,h).to(torch.bfloat16)
        return (q,key,value,g,beta)
    with torch.inference_mode():
        for b,t,h,k,v in [(1,1,1,1,4),(2,2,3,7,12),(1,31,2,128,128),
                           (1,115,16,128,128),(1,127,3,128,128),(1,128,2,128,128),
                           (1,129,2,128,128),(1,256,2,128,128),(1,257,2,128,128),(1,512,2,128,128)]:
            base_args=inputs(b,t,h,k,v)
            for norm in (False,True):
                # Unnormalized random keys can form a poorly conditioned delta
                # system; use bounded keys for that valid equation comparison.
                args=base_args if norm else (base_args[0]*.03,base_args[1]*.03,*base_args[2:])
                copies=[x.clone() for x in args]
                ref=qwen.torch_recurrent_gated_delta_rule(*args,use_qk_l2norm_in_kernel=norm)[0]
                chunk=chunk_delta_prefill(*args,chunk_size=128,use_qk_l2norm_in_kernel=norm)[0]
                for tile in (16,32,64,128):
                    for fused in (0,1,2,3,4,5):
                        fn=make_scan(lambda *a,**k:None,tile=tile,fused=fused)
                        out=fn(*args,use_qk_l2norm_in_kernel=norm)[0]
                        assert fn.counts=={'optimized':1,'fallback':0}
                        rel=(torch.linalg.vector_norm(out.float()-ref.float())/torch.linalg.vector_norm(ref.float()).clamp_min(1e-20)).item()
                        assert torch.isfinite(out).all() and rel<.01,(b,t,h,k,v,norm,tile,fused,rel)
                        assert all(torch.equal(x,y) for x,y in zip(args,copies))
                        checks.append({'shape':[b,t,h,k,v],'norm':norm,'tile':tile,'fused':fused,'relative_l2_vs_official_recurrent':rel,
                                       'max_abs_vs_chunk':(out.float()-chunk.float()).abs().max().item()})
        args=inputs(1,31,2,128,128);later=[x.clone() for x in args]
        cut=13
        for x in later[:3]:x[:,cut:]+=2
        later[3][:,cut:]-=1;later[4][:,cut:]=0
        out=native(*args,use_qk_l2norm_in_kernel=True)[0]
        changed=native(*later,use_qk_l2norm_in_kernel=True)[0]
        assert torch.equal(out[:,:cut],changed[:,:cut])
        # Different requests cannot retain recurrent state or scratch contents.
        native(*inputs(1,2,1,128,128),use_qk_l2norm_in_kernel=True)
        assert torch.equal(out,native(*args,use_qk_l2norm_in_kernel=True)[0])
        calls=[];marker=object()
        def fallback(*args,**kwargs):calls.append(kwargs);return marker
        fn=make_scan(fallback)
        for kw in ({'initial_state':torch.zeros(1)}, {'output_final_state':True},
                   {'cu_seqlens':torch.tensor([0,31])}, {'unsupported_future_option':True}):
            assert fn(*args,**kw) is marker
        assert fn(*inputs(1,513,1,128,128)) is marker
        compact_fallback=make_scan(fallback,'compact')
        assert compact_fallback(*inputs(1,129,1,128,128)) is marker
        assert fn(*(x.float() for x in args)) is marker
        assert fn(*inputs(1,0,1,128,128)) is marker
        assert fn(*inputs(1,2,1,128,7)) is marker
        assert fn(*inputs(1,2,1,129,128)) is marker
    result={'passed':True,'equation_checks':checks,'causal_prefix_bitwise_equal':True,'inputs_unchanged':True,
            'requests_isolated':True,'fallback_calls':len(calls),'scope':'Finite bounded random tensors vs installed official public recurrent implementation, both norm settings, short sequence boundaries, scalar key dimensions, value tails, noncontiguous inputs. Long/unsupported/stateful contracts delegate unchanged.'}
    Path('/work/logs/unit-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SCAN_UNITS',len(checks),'passed',flush=True)
    return result
