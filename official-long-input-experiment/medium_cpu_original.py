"""Scoped actual pre-fusion CPU methods, preserving other optimized domains.

References come from this pinned owner's wrappers, not raw Torch BF16. Save and
restore exact method objects so private115/615 method-identity guards survive.
"""
from contextlib import contextmanager
from types import MethodType
import inspect
import torch


def references(backbone):
    found=[]
    for name,module in backbone.named_modules():
        fn=module.forward;function=getattr(fn,'__func__',fn);source=getattr(function,'__module__','');kind=None
        if source=='lut_ops' and type(module).__name__ in ('Qwen3_5RMSNorm','Qwen3_5RMSNormGated'):kind='norm'
        elif source=='small_linear' and isinstance(module,torch.nn.Linear):kind='small'
        elif source=='residual' and type(module).__name__=='Qwen3_5DecoderLayer':kind='residual'
        if kind is None:continue
        parameter=inspect.signature(fn).parameters.get('_old')
        if parameter is not None and parameter.default is not inspect.Parameter.empty and callable(parameter.default):
            found.append((name,module,fn,parameter.default,kind))
    return found

@contextmanager
def restored_methods(states):
    saved=[]
    try:
        for _,module,_,reference,_ in states:
            saved.append((module,module.forward));module.forward=reference
        yield
    finally:
        for module,original in reversed(saved):module.forward=original

class MediumCpuOriginal:
    def __init__(self,owner):
        self.model=owner.model;self.original=self.model.logits;self.states=references(self.model.backbone)
        if not self.states:raise ValueError('No captured pre-fusion CPU methods in this owner')
        self.closed=False;self.active=False
        self.counts={'scopes':0,'bypass':0,'guard_bypass':0,'methods':{},'batch_shapes':{}}
        for *_,kind in self.states:self.counts['methods'][kind]=self.counts['methods'].get(kind,0)+1
        def scope(model,sequences,counts):
            b=len(sequences);t=max(map(len,sequences),default=0)
            good=(not self.closed and not self.active and not torch.is_grad_enabled() and 0<b<=8 and 512<t<=2048 and 2048<b*t<=32768)
            if not good:self.counts['bypass']+=1;return self.original(sequences,counts)
            if any(m.forward!=native or m.training or m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for _,m,native,_,_ in self.states):
                self.counts['guard_bypass']+=1;return self.original(sequences,counts)
            self.active=True;self.counts['scopes']+=1;k=f'{b}x{t}';self.counts['batch_shapes'][k]=self.counts['batch_shapes'].get(k,0)+1
            try:
                with restored_methods(self.states):return self.original(sequences,counts)
            finally:self.active=False
        self.wrapper=MethodType(scope,self.model);self.model.logits=self.wrapper
    def close(self):
        if self.closed:return
        if self.active:raise RuntimeError('Cannot close an active original CPU scope')
        self.closed=True
        if self.model.logits==self.wrapper:self.model.logits=self.original

def install(owner):
    owner.medium_cpu_original=MediumCpuOriginal(owner)
    return owner.medium_cpu_original.close
