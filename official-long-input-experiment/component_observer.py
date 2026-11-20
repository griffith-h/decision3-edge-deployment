"""Compare first component outputs on identical inputs without changing results."""
import json

class ObservationError(Exception):pass

class Observer:
    def __init__(self,owner,out):
        import torch,long_kernels as lk
        from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
        self.owner=owner;self.out=out;self.mode=None;self.records=[];self.done=set();self.restores=[];self.closed=False
        def replace(obj,name,fn):
            old=getattr(obj,name);setattr(obj,name,fn)
            self.restores.append(lambda o=obj,n=name,v=old:setattr(o,n,v))
        old_projection=owner.projections.run
        old_mlp=owner.native_mlp.run
        old_attention=lk.attention
        old_scan=qwen.torch_chunk_gated_delta_rule
        activations={id(s[2][0]):s[-1] for s in owner.long_mlp.states}
        def projection(members,x,key):
            result=old_projection(members,x,key)
            if self.mode=='only_projection' and 'projection' not in self.done:
                def observe():
                    for m in members:
                        name=next(n for n,v in owner.projections.modules.items() if v is m)
                        expected=m._bp_original(x)
                        self.record('projection',result[id(m)],expected,{'module':name,'input_shape':list(x.shape)})
                        import torch.nn.functional as f
                        quantized=f.linear(x.half().float(),m.weight.half().float(),m.bias.float() if m.bias is not None else None).to(x.dtype)
                        self.record('projection_fp16_conversion_cpu',quantized,expected,{'module':name,'reference':'CPU FP32 multiply after FP16 input/weight conversion; diagnostic only'})
                        flat=x.reshape(-1,x.shape[-1]);chunks=torch.cat([m._bp_original(part) for part in flat.split(512)],0).reshape(expected.shape)
                        self.record('projection_cpu_row512',chunks,expected,{'module':name,'reference':'Original CPU projection at512-row geometry; diagnostic only'})
                self.protected(observe);self.done.add('projection')
            return result
        replace(owner.projections,'run',projection)
        def mlp(members,x):
            result=old_mlp(members,x)
            if self.mode=='only_mlp' and 'mlp' not in self.done:
                def observe():
                    gate,up,down=members
                    expected=down._bp_original(activations[id(gate)](gate._bp_original(x))*up._bp_original(x))
                    self.record('mlp',result,expected,{'input_shape':list(x.shape)})
                self.protected(observe);self.done.add('mlp')
            return result
        replace(owner.native_mlp,'run',mlp)
        def attention(q,k,v,mask,scale):
            result=old_attention(q,k,v,mask,scale)
            if self.mode=='only_attention' and 'attention' not in self.done:
                def observe():
                    import torch.nn.functional as f
                    kk=k.repeat_interleave(q.shape[1]//k.shape[1],1);vv=v.repeat_interleave(q.shape[1]//v.shape[1],1)
                    expected=f.scaled_dot_product_attention(q,kk,vv,attn_mask=mask,dropout_p=0.,is_causal=False,scale=scale).transpose(1,2)
                    self.record('attention',result,expected,{'query_shape':list(q.shape),'reference':'Original BF16 PyTorch SDPA with repeated KV'})
                    full=f.scaled_dot_product_attention(q.float(),kk.float(),vv.float(),attn_mask=mask,dropout_p=0.,is_causal=False,scale=scale).transpose(1,2).bfloat16()
                    self.record('attention_fp32_alternative',result,full,{'reference':'FP32 alternative, diagnostic only'})
                self.protected(observe);self.done.add('attention')
            return result
        replace(lk,'attention',attention)
        def scan(*args,**kwargs):
            result=old_scan(*args,**kwargs)
            if self.mode=='only_scan' and 'scan' not in self.done:
                def observe():
                    expected=owner.long.original_scan(*args,**kwargs)
                    self.record('scan',result[0],expected[0],{'query_shape':list(args[0].shape),'reference':'Original current CPU scan on identical tensors'})
                self.protected(observe);self.done.add('scan')
            return result
        replace(qwen,'torch_chunk_gated_delta_rule',scan)
    def protected(self,fn):
        try:fn()
        except Exception as error:raise ObservationError('Component observation failed: '+str(error)) from error
    def record(self,kind,actual,expected,metadata):
        import torch
        assert actual.shape==expected.shape
        a=actual.detach().float();b=expected.detach().float();delta=(a-b).abs()
        assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        row={'kind':kind,'shape':list(actual.shape),'actual_dtype':str(actual.dtype),'reference_dtype':str(expected.dtype),
             'max_absolute_error':float(delta.max()),'mismatch_fraction':float((a!=b).float().mean()),
             'relative_l2':float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(b).clamp_min(1e-30)),**metadata}
        if actual.dtype==expected.dtype==torch.bfloat16:
            same_nonzero=(a!=0)&(b!=0)&((a<0)==(b<0))
            ulps=(actual.contiguous().view(torch.int16).to(torch.int32)-expected.contiguous().view(torch.int16).to(torch.int32)).abs()[same_nonzero]
            row['same_sign_nonzero_ulp_counts']={str(n):int((ulps==n).sum()) for n in (0,1,2)}
            row['same_sign_nonzero_ulp_gt2']=int((ulps>2).sum())
        self.records.append(row);self.save()
    def save(self):
        (self.out/'component-observations.json').write_text(json.dumps({'closed':self.closed,'observed':sorted(self.done),
            'scope':'Identical real component inputs; original outputs returned unchanged; not a benchmark score','records':self.records})+'\n')
    def close(self):
        if self.closed:return
        self.closed=True
        for restore in reversed(self.restores):restore()
        self.save()
