"""Development tiling of the repository current cached-FP32 CPU linear path.

Only guarded medium-domain row-independent linears change. B/T, padding,
attention and scan remain original; weights are temporary FP32 views of the
same BF16 parameters, never modified. No long/protected-path change.
"""
import torch
import torch.nn.functional as F

class MediumDense:
    def __init__(self,owner):
        self.owner=owner;self.engine=owner.engine;self.native=owner.native_mlp;self.closed=False
        self.group_old=self.engine.group;self.execute_old=self.engine.execute;self.mlp_old=self.native.run
        self.activations={id(state[2][0]):state[-1] for state in owner.long_mlp.states}
        self.counts={'groups':0,'singles':0,'mlp_chunks':0,'linear_calls':0,'maximum_weight_bytes':0,'errors':[]}
        def execute(module,x,prepared=False,output=None):
            if not self.active():return self.execute_old(module,x,prepared,output)
            if prepared or output is not None:raise ValueError('Dense experiment requires private original input/output')
            self.counts['singles']+=1;return self.linear(module,x)
        def group(modules,x):
            if not self.active():return self.group_old(modules,x)
            self.counts['groups']+=1
            return {id(m):self.linear(m,x) for m in modules}
        def mlp(members,x):
            if not self.active() or not owner.long_mlp.active:return self.mlp_old(members,x)
            gate,up,down=members;product=self.activations[id(gate)](self.linear(gate,x))*self.linear(up,x)
            value=self.linear(down,product);self.counts['mlp_chunks']+=1;return value
        self.execute=execute;self.group=group;self.mlp=mlp
        self.engine.execute=execute;self.engine.group=group;self.native.run=mlp
    def active(self):
        return not self.closed and getattr(getattr(self.owner,'medium_precision',None),'active',False) and self.owner.long_projection.active
    def linear(self,module,x):
        if torch.is_grad_enabled() or x.dtype!=torch.bfloat16 or x.device.type!='cpu' or module.bias is not None:
            raise ValueError('Unsupported dense experiment linear')
        shape=(*x.shape[:-1],module.out_features);flat=x.reshape(-1,x.shape[-1]);out=torch.empty(shape,dtype=x.dtype,device=x.device)
        weight=getattr(module,'_cpu_fp32_weight',None)
        if weight is None or weight.device.type!='cpu' or weight.dtype!=torch.float32 or weight.shape!=module.weight.shape:
            raise ValueError('Missing repository current FP32 CPU weight buffer')
        self.counts['maximum_weight_bytes']=max(self.counts['maximum_weight_bytes'],weight.numel()*4)
        target=out.reshape(-1,module.out_features)
        for start in range(0,len(flat),512):target[start:start+512].copy_(F.linear(flat[start:start+512].float(),weight).bfloat16())
        self.counts['linear_calls']+=1;return out
    def close(self):
        if self.closed:return
        if self.active():raise RuntimeError('Cannot close active dense experiment')
        self.closed=True
        if self.engine.execute is self.execute:self.engine.execute=self.execute_old
        if self.engine.group is self.group:self.engine.group=self.group_old
        if self.native.run is self.mlp:self.native.run=self.mlp_old

def install(owner):
    owner.medium_dense=MediumDense(owner)
    return owner.medium_dense.close
