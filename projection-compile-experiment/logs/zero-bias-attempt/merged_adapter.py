"""Same-input output-column packing; original model parameters are never assigned."""
import threading
import time
import torch
from safe_engine import SafeEngine, OutputRangeError

class PackedEngine(SafeEngine):
    def __init__(self, mode):
        super().__init__(fusion=True,parallel=True)
        assert mode in ('merge','balanced')
        self.mode=mode;self.packed={};self.layout=[]
        self.stats.update(packed_groups=0,packed_range_fallbacks=0)

    def pack(self,modules):
        key=tuple(id(m) for m in modules)
        if key in self.packed:return self.packed[key]
        sizes=[m.out_features for m in modules]
        # MLP gate/up are already equal. Delta's qkv/z may need rebalancing.
        if len(modules)==2 and (self.mode=='merge' or sizes[0]==sizes[1]):return None
        if len(modules) not in (2,3):return None
        if len({m.in_features for m in modules})!=1:return None
        if any(m.weight.dtype!=torch.bfloat16 for m in modules):return None
        if self.mode=='merge':groups=[[modules[0]],list(modules[1:])]
        else:groups=None
        with torch.no_grad():
            weight=torch.cat([m.weight.detach() for m in modules],dim=0)
            bias=torch.cat([m.bias.detach().float() if m.bias is not None else torch.zeros(m.out_features) for m in modules])
            split=sizes[0] if groups else (weight.shape[0]//2//16)*16
            widths=[split,weight.shape[0]-split]
            if min(widths)<=0 or any(n%16 for n in widths):return None
            packed=[];offset=0
            for i,n in enumerate(widths):
                # A lightweight private module owns an exact slice of packed parameters.
                m=torch.nn.Linear.__new__(torch.nn.Linear);torch.nn.Module.__init__(m)
                m.in_features=weight.shape[1];m.out_features=n
                m.weight=torch.nn.Parameter(weight[offset:offset+n].contiguous(),requires_grad=False)
                m.bias=torch.nn.Parameter(bias[offset:offset+n].contiguous(),requires_grad=False)
                w=m.weight.float();b=m.bias.float()
                m._bp_original=lambda x,w=w,b=b:torch.nn.functional.linear(x.float(),w,b).bfloat16()
                m._bp_handle=None;self.core[id(m)]=i+1;self.bucket[id(m)]=i
                packed.append(m);offset+=n
        entry=(packed,sizes);self.packed[key]=entry
        self.layout.append({'original':sizes,'packed':widths,'k':weight.shape[1],'cores':[1,2]})
        return entry

    def group(self,modules,x):
        entry=self.pack(modules)
        if entry is None:return super().group(modules,x)
        packed,sizes=entry
        # Use the base execute to expose -3; fallback the entire original group,
        # preserving original CPU projection rounding instead of packed CPU GEMM.
        for m in packed:self.ensure(m)
        rows=x.numel()//x.shape[-1];inp=x.contiguous()
        self.status(self.lib.mm_prepare_bf16(self.ensure(packed[0]),rows,inp.data_ptr()))
        self.status(self.lib.mm_copy_prepared(self.ensure(packed[1]),self.ensure(packed[0]),rows))
        from adapter import Engine
        jobs=[self.pool.submit(Engine.execute,self,m,x,True) for m in packed]
        outputs=[];error=None;range_failed=False
        for job in jobs:
            try:outputs.append(job.result())
            except OutputRangeError:range_failed=True
            except Exception as exc:
                if error is None:error=exc
        # No future may still own a native input/output buffer at this point.
        if error is not None:raise error
        self.stats['packed_groups']+=1;self.stats['shared_input_prepares']+=1;self.stats['parallel_jobs']+=2
        if range_failed:
            self.stats['packed_range_fallbacks']+=1;self.cpu_fallbacks+=len(modules)
            return {id(m):m._bp_original(x) for m in modules}
        joined=torch.cat(outputs,dim=-1)
        return {id(m):part.contiguous() for m,part in zip(modules,joined.split(sizes,dim=-1))}

    def close(self):
        super().close();self.packed.clear()
