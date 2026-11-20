"""Experimental long-MLP tiling through the existing BF16-boundary native kernel.

Only token-wise MLP rows are tiled. Attention, masks, positions and scan state
are untouched. All chunks stay private; a failure replays the whole old MLP.
"""
from types import MethodType
import torch


class BoundedMlp:
    def __init__(self,owner,max_rows=32768):
        self.owner=owner;self.model=owner.model;self.native=owner.native_mlp
        self.max_rows=max_rows;self.closed=False;self.active=False;self.disabled=set()
        self.original=self.model.logits;self.states=[]
        self.counts={'layers':0,'chunks':0,'padded_rows':0,'range_fallbacks':0,'runtime_errors':[],
                     'guard_fallbacks':0,'shapes':{}}
        for layer in self.model.backbone.modules():
            if type(layer).__name__!='Qwen3_5MLP':continue
            members=(layer.gate_proj,layer.up_proj,layer.down_proj)
            params=tuple((m,m.weight,m.weight._version,m.bias) for m in members)
            self.states.append((layer,layer.forward,members,params,layer.act_fn))
        def scope(model,sequences,counts):
            rows=len(sequences)*max(map(len,sequences),default=0)
            if self.closed or self.active or not 2048<rows<=self.max_rows:
                return self.original(sequences,counts)
            changed=[];self.active=True
            try:
                for layer,old,members,params,activation in self.states:
                    if layer.forward!=old:continue
                    def forward(x,_layer=layer,_old=old,_members=members,_params=params,_activation=activation):
                        return self.forward(_layer,_old,_members,_params,_activation,x)
                    layer.forward=forward;changed.append((layer,old))
                return self.original(sequences,counts)
            finally:
                for layer,old in reversed(changed):layer.forward=old
                self.active=False
        self.wrapper=MethodType(scope,self.model);self.model.logits=self.wrapper

    def good(self,layer,members,params,activation,x):
        return (not torch.is_grad_enabled() and not layer.training and id(layer) not in self.disabled
                and isinstance(x,torch.Tensor) and x.device.type=='cpu' and x.dtype==torch.bfloat16
                and x.layout==torch.strided and x.ndim==3 and x.shape[-1]==1024
                and 0<x.shape[0]<=8 and 2048<x.shape[0]*x.shape[1]<=self.max_rows
                and layer.act_fn is activation and type(activation).__name__ in ('SiLU','SiLUActivation')
                and not activation._forward_hooks and not activation._forward_pre_hooks and not activation._backward_hooks
                and not layer._forward_hooks and not layer._forward_pre_hooks and not layer._backward_hooks
                and all(getattr(layer,name) is module for name,module in zip(('gate_proj','up_proj','down_proj'),members))
                and all(m is original and m.weight is weight and weight._version==version and m.bias is bias
                        and bias is None and weight.dtype==torch.bfloat16 and weight.device.type=='cpu'
                        and not m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                        for m,(original,weight,version,bias) in zip(members,params)))

    def forward(self,layer,old,members,params,activation,x):
        if not self.good(layer,members,params,activation,x):
            self.counts['guard_fallbacks']+=1;return old(x)
        shape=str(tuple(x.shape));self.counts['shapes'][shape]=self.counts['shapes'].get(shape,0)+1
        try:
            flat=x.reshape(-1,1024);result=torch.empty_like(flat)
            for start in range(0,flat.shape[0],615):
                used=min(615,flat.shape[0]-start)
                if used==615:part=flat[start:start+615].reshape(1,615,1024)
                else:
                    part=torch.zeros((1,615,1024),dtype=x.dtype,device=x.device)
                    part[0,:used].copy_(flat[start:start+used]);self.counts['padded_rows']+=615-used
                value=self.native.run(members,part)
                result[start:start+used].copy_(value.reshape(615,1024)[:used]);self.counts['chunks']+=1
        except ValueError:
            self.counts['range_fallbacks']+=1;return old(x)
        except RuntimeError as exc:
            self.counts['runtime_errors'].append(str(exc));self.disabled.add(id(layer));return old(x)
        self.counts['layers']+=1
        return result.reshape(x.shape)

    def close(self):
        if self.closed:return
        if self.active:raise RuntimeError('Cannot close an active MLP scope')
        self.closed=True
        if self.model.logits==self.wrapper:self.model.logits=self.original


def install(owner):
    owner.long_mlp=BoundedMlp(owner)
    return owner.long_mlp.close
