import baseline,torch,json
from pathlib import Path
from chunk_blas2 import install
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
old=qwen.torch_chunk_gated_delta_rule;seen=[]
def original(*args,**kw):seen.append(kw);return 'original'
qwen.torch_chunk_gated_delta_rule=original;restore,counts=install(32,64)
try:
    fn=qwen.torch_chunk_gated_delta_rule;q=torch.randn(1,9,2,8).bfloat16();k=q.clone();v=q.clone();g=-torch.rand(1,9,2);b=torch.rand(1,9,2).bfloat16()
    with torch.inference_mode():
        for kw in ({'initial_state':torch.zeros(1,2,8,8)},{'output_final_state':True},{'cu_seqlens':torch.tensor([0,9])}):assert fn(q,k,v,g,b,**kw)=='original'
        for n in (0,513):
            x=torch.empty(1,n,2,8,dtype=torch.bfloat16);z=torch.zeros(1,n,2);assert fn(x,x,x,z,z)=='original'
        assert fn(q.float(),k.float(),v.float(),g,b)=='original'
        saved=q.clone();out,state=fn(q,k,v,g,b,use_qk_l2norm_in_kernel=True,use_cache=False)
        assert out.shape==v.shape and out.dtype==v.dtype and state is None and torch.equal(q,saved)
    with torch.enable_grad():assert fn(q,k,v,g,b)=='original'
finally:restore();assert qwen.torch_chunk_gated_delta_rule is original;qwen.torch_chunk_gated_delta_rule=old
Path('/work/logs/chunk-wrapper-unit.json').write_text(json.dumps({'passed':True,'counts':counts,'original_calls':len(seen)},indent=2)+'\n');print('PASS chunk wrapper',flush=True)
