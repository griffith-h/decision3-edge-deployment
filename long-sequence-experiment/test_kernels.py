import baseline
import torch, json, math
from pathlib import Path
import long_kernels as lk
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

torch.set_num_threads(3)
torch.manual_seed(1819)
P=Path('/experiment'); result={'attention':[], 'scan':[], 'guards':[]}

def scan_ref(xs):
    q,k,v,d,b=[x.double() for x in xs]
    state=torch.zeros(q.shape[0],q.shape[1],q.shape[3],v.shape[3],dtype=torch.float64)
    out=[]
    for i in range(q.shape[2]):
        state=state*d[:,:,i,None,None]
        delta=(v[:,:,i]-(state*k[:,:,i,:,None]).sum(-2))*b[:,:,i,None]
        state+=k[:,:,i,:,None]*delta[:,:,:,None].transpose(-2,-1)
        out.append((state*q[:,:,i,:,None]).sum(-2))
    return torch.stack(out,2).bfloat16()

with torch.inference_mode():
    for b,h,kh,t,d in [(1,2,1,512,16),(1,2,1,513,16),(2,4,2,615,32),(1,2,1,1024,64),(1,2,1,1025,128),(1,2,1,2048,256)]:
        # Head/sequence transposes exercise non-contiguous outer strides.
        q=torch.randn(b,t,h,d).bfloat16().transpose(1,2)
        k=torch.randn(b,t,kh,d).bfloat16().transpose(1,2);v=torch.randn_like(k)
        mask=torch.ones(b,1,1,t,dtype=torch.bool);mask[...,13::19]=False
        saved=[x.clone() for x in (q,k,v,mask)]
        out=lk.attention(q,k,v,mask,d**-.5)
        krep=k.repeat_interleave(h//kh,1).float();vrep=v.repeat_interleave(h//kh,1).float()
        ref=torch.nn.functional.scaled_dot_product_attention(q.float(),krep,vrep,attn_mask=mask,scale=d**-.5).transpose(1,2).bfloat16()
        delta=(out.float()-ref.float()).abs().max().item()
        assert torch.allclose(out.float(),ref.float(),atol=.002,rtol=.008),(t,delta)
        assert all(torch.equal(a,z) for a,z in zip((q,k,v,mask),saved))
        result['attention'].append({'shape':[b,h,kh,t,d],'max_abs':delta})
    # Noncausal: the final value must influence the first query across 512.
    q=torch.zeros(1,1,513,4,dtype=torch.bfloat16);mask=torch.ones(1,1,1,513,dtype=torch.bool)
    v=torch.zeros_like(q);v[:,:,512]=128
    out=lk.attention(q,q,v,mask,.5)
    assert out[0,0,0,0]>0
    mask[...,512]=False
    assert torch.count_nonzero(lk.attention(q,q,v,mask,.5))==0
    result['noncausal_future_and_mask']=True

    for b,h,t,kd,vd in [(1,1,513,4,4),(2,2,615,12,20),(1,2,1024,32,16),(1,1,1025,128,128),(1,1,2048,64,32)]:
        q,k=[torch.nn.functional.normalize(torch.randn(b,h,t,kd),dim=-1) for _ in range(2)];q*=kd**-.5
        v=torch.randn(b,h,t,vd);decay=torch.rand(b,h,t)*.1+.89;beta=torch.rand(b,h,t)
        decay[:,:,509]=0;decay[:,:,510]=1e-35;beta[:,:,511]=0
        xs=(q,k,v,decay,beta);saved=[x.clone() for x in xs]
        out=lk.scan_prepared(*xs);ref=scan_ref(xs);delta=(out.float()-ref.float()).abs().max().item()
        assert torch.allclose(out.float(),ref.float(),atol=.001,rtol=.008),(t,delta)
        # Prefixes at 512/513 and partial block boundaries retain the same state.
        for n in (512,513,t-1):
            prefix=tuple(x[:,:,:n].contiguous() for x in xs)
            assert torch.allclose(out[:,:,:n].float(),lk.scan_prepared(*prefix).float(),atol=.001,rtol=.008),(t,n)
        assert all(torch.equal(a,z) for a,z in zip(xs,saved))
        result['scan'].append({'shape':[b,h,t,kd,vd],'max_abs':delta})
    # A state impulse before 512 survives after 512: zero-state splitting fails.
    q=torch.ones(1,1,615,4)*.25;k=torch.zeros_like(q);k[:,:,0]=.5
    v=torch.zeros(1,1,615,4);v[:,:,0]=1;decay=torch.ones(1,1,615);beta=torch.ones_like(decay)
    full=lk.scan_prepared(q,k,v,decay,beta)
    tail=lk.scan_prepared(*(x[:,:,512:].contiguous() for x in (q,k,v,decay,beta)))
    assert full[:,:,512:].abs().max()>0 and tail.abs().max()==0
    result['scan_state_crosses_512']=True

    interface=qwen.ALL_ATTENTION_FUNCTIONS;oldatt=interface['sdpa'];oldscan=qwen.torch_chunk_gated_delta_rule
    sentinel=(object(),object());calls=[]
    def spy(*args,**kwargs):calls.append((args,kwargs));return sentinel
    interface.register('sdpa',spy);qwen.torch_chunk_gated_delta_rule=spy
    kernels=lk.Kernels();kernels.enabled={'attention','scan'}
    module=object.__new__(qwen.Qwen3_5Attention);torch.nn.Module.__init__(module);module.scaling=.25;module.eval()
    q=torch.randn(1,2,513,16).bfloat16();k=torch.randn(1,1,513,16).bfloat16();v=torch.randn_like(k);mask=torch.ones(1,1,1,513,dtype=torch.bool)
    try:
        assert kernels.attention(module,q,k,v,mask)[0].shape==(1,513,2,16)
        for tag,m,kw in [('none',None,{}),('all-masked',mask&False,{}),('causal',mask,{'is_causal':True}),('dropout',mask,{'dropout':.1}),('cache',mask,{'use_cache':True}),('unknown',mask,{'unknown':1}),('scale-none',mask,{'scaling':None}),('scale-nan',mask,{'scaling':float('nan')})]:
            assert kernels.attention(module,q,k,v,m,**kw) is sentinel
            assert calls[-1][0][4] is m and calls[-1][1].keys()==kw.keys() and all(calls[-1][1][k] is value for k,value in kw.items())
            result['guards'].append('attention-'+tag)
        for t in (512,2049):
            z=torch.zeros(1,1,t,16,dtype=torch.bfloat16);m=torch.ones(1,1,1,t,dtype=torch.bool)
            assert kernels.attention(module,z,z,z,m) is sentinel;result['guards'].append('attention-t'+str(t))
        module.train();assert kernels.attention(module,q,k,v,mask) is sentinel;module.eval()
        with torch.enable_grad():assert kernels.attention(module,q,k,v,mask) is sentinel
        rawq=torch.randn(2,513,2,12).bfloat16();rawk=torch.randn_like(rawq);rawv=torch.randn(2,513,2,20).bfloat16()
        g=-torch.rand(2,513,2);b=torch.rand(2,513,2)
        assert kernels.scan(rawq,rawk,rawv,g,b,use_qk_l2norm_in_kernel=True)[0].shape==rawv.shape
        for tag,kw in [('initial',{'initial_state':torch.zeros(1)}),('final',{'output_final_state':True}),('unknown',{'unknown':1}),('cache',{'use_cache':True})]:
            assert kernels.scan(rawq,rawk,rawv,g,b,**kw) is sentinel
            for key,value in kw.items():assert calls[-1][1][key] is value
            result['guards'].append('scan-'+tag)
        for t in (512,2049):
            z=torch.zeros(1,t,1,4,dtype=torch.bfloat16);a=torch.zeros(1,t,1)
            assert kernels.scan(z,z,z,a,a) is sentinel;result['guards'].append('scan-t'+str(t))
        with torch.enable_grad():assert kernels.scan(rawq,rawk,rawv,g,b) is sentinel
        # Inject rejection after the native output would have been private.
        oldfn=lk.scan_prepared
        def fail(*a):raise ValueError('injected whole-call failure')
        lk.scan_prepared=fail
        assert kernels.scan(rawq,rawk,rawv,g,b) is sentinel and 'scan' in kernels.disabled
        lk.scan_prepared=oldfn
        assert kernels.scan(rawq,rawk,rawv,g,b) is sentinel
        bad=q.clone();bad[0,0,0,0]=float('nan')
        assert kernels.attention(module,bad,k,v,mask) is sentinel and 'attention' in kernels.disabled
        result['guards']+=['scan-whole-call-fallback-and-disable','attention-nan-fallback-and-disable']
        kernels.close();kernels.close()
        assert interface['sdpa'] is spy and qwen.torch_chunk_gated_delta_rule is spy
    finally:
        kernels.close();interface.register('sdpa',oldatt);qwen.torch_chunk_gated_delta_rule=oldscan
result['passed']=True
(P/'logs/unit.json').write_text(json.dumps(result,indent=2)+'\n')
print('LONG KERNEL UNIT PASS',result,flush=True)
