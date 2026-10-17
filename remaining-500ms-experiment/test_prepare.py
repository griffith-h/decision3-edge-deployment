import sys,json
sys.path[:0]=['/base','/scan','/release']
import torch
from prepared_scan import prepare,install
from scan import recurrent
from pathlib import Path
torch.set_num_threads(3);torch.manual_seed(315)
rows=[]
with torch.inference_mode():
    for batch,length,heads,kd,vd in [(1,1,1,4,4),(2,3,2,12,8),(1,115,16,128,128),(1,129,2,64,32),(1,512,1,128,128)]:
        combined=torch.randn(batch,length,heads,kd*2+vd).bfloat16();q,k,v=combined.split([kd,kd,vd],dim=-1)
        saved=combined.clone()
        for normalize in (False,True):
            actual=prepare(q,k,v,normalize)
            qq,kk,vv=[x.transpose(1,2).float().contiguous() for x in (q,k,v)]
            if normalize:
                qq=qq*torch.rsqrt((qq*qq).sum(-1,keepdim=True)+1e-6);kk=kk*torch.rsqrt((kk*kk).sum(-1,keepdim=True)+1e-6)
            qq=qq*kd**-.5
            errs=[]
            for a,b in zip(actual,(qq,kk,vv)):
                assert torch.allclose(a,b,atol=2e-7,rtol=2e-6),(a-b).abs().max();errs.append((a-b).abs().max().item())
            assert torch.equal(combined,saved);rows.append({'shape':[batch,length,heads,kd,vd],'normalize':normalize,'errors':errs})
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    old=qwen.torch_chunk_gated_delta_rule;calls=[]
    def sentinel(*args,**kwargs):calls.append(kwargs);return 'fallback'
    qwen.torch_chunk_gated_delta_rule=sentinel;restore,counts=install();fn=qwen.torch_chunk_gated_delta_rule
    q=torch.randn(1,3,2,8).bfloat16();k=q.clone();v=q.clone();g=-torch.rand(1,3,2);beta=torch.rand(1,3,2).bfloat16()
    for kw in ({'initial_state':torch.zeros(1,2,8,8)},{'output_final_state':True},{'cu_seqlens':torch.tensor([0,3])}):assert fn(q,k,v,g,beta,**kw)=='fallback'
    for n in (0,513):
        qq=torch.empty(1,n,2,8,dtype=torch.bfloat16);gg=torch.empty(1,n,2)
        assert fn(qq,qq,qq,gg,gg)=='fallback'
    assert fn(q.float(),k.float(),v.float(),g,beta)=='fallback'
    assert fn(q.transpose(2,3),k.transpose(2,3),v.transpose(2,3),torch.zeros(1,3,8),torch.zeros(1,3,8))=='fallback'
    out,_=fn(q,k,v,g,beta,use_qk_l2norm_in_kernel=True,use_cache=False)
    ref,_=recurrent(q,k,v,g,beta,use_qk_l2norm_in_kernel=True,tile=16,fused=5)
    assert torch.allclose(out.float(),ref.float(),atol=.002,rtol=.01)
    restore();assert qwen.torch_chunk_gated_delta_rule is sentinel;qwen.torch_chunk_gated_delta_rule=old
Path('/work/logs/prepare-unit.json').write_text(json.dumps({'passed':True,'cases':rows,'counts':counts,'fallback_calls':len(calls)},indent=2)+'\n')
print('PASS preparation',len(rows),counts,flush=True)
