"""Retire derived FP32 CPU caches of private packed modules, not model weights.

The fixed PackedEngine packs BF16 parameters and also captures FP32 copies in
private fallback lambdas. Rebuild those derived copies only if that private CPU
fallback is actually called. Original model modules/guards remain untouched.
"""
import weakref
import torch
import torch.nn.functional as F

class PackedCachePolicy:
    def __init__(self,owner):
        self.engine=owner.engine;self.previous=self.engine.pack;self.closed=False
        self.counts={'retired_modules':0,'derived_fp32_bytes_released':0,'private_cpu_calls':0,'guard_bypasses':0}
        self.seen=weakref.WeakSet()
        def pack(modules):
            entry=self.previous(modules)
            if not self.closed and entry is not None:self.retire(entry[0])
            return entry
        self.wrapper=pack
        # Only the private modules in this owner-owned bank are eligible.
        for modules,_ in self.engine.packed.values():self.retire(modules)
        self.engine.pack=pack
    def retire(self,modules):
        for module in modules:
            if module in self.seen:continue
            original=getattr(module,'_bp_original',None);defaults=getattr(original,'__defaults__',None)
            good=(type(module) is torch.nn.Linear and not module.weight.requires_grad and defaults is not None and len(defaults)==2
                and not getattr(original,'__closure__',None)
                and isinstance(defaults[0],torch.Tensor) and defaults[0].device.type=='cpu' and defaults[0].dtype==torch.float32
                and defaults[0].shape==module.weight.shape and module.weight.device.type=='cpu' and module.weight.dtype==torch.bfloat16
                and (defaults[1] is None and module.bias is None or isinstance(defaults[1],torch.Tensor)
                     and module.bias is not None and defaults[1].shape==module.bias.shape and defaults[1].dtype==torch.float32))
            if not good:self.counts['guard_bypasses']+=1;continue
            released=sum(x.numel()*x.element_size() for x in defaults if isinstance(x,torch.Tensor))
            reference=weakref.ref(module);counts=self.counts
            def private_cpu(x,_ref=reference,_counts=counts):
                current=_ref()
                if current is None:raise RuntimeError('Private packed CPU module no longer exists')
                _counts['private_cpu_calls']+=1
                weight=current.weight.float();bias=None if current.bias is None else current.bias.float()
                return F.linear(x.float(),weight,bias).bfloat16()
            module._bp_original=private_cpu;self.seen.add(module)
            self.counts['retired_modules']+=1;self.counts['derived_fp32_bytes_released']+=released
            # Do not retain old lambdas: their defaults own the redundant caches.
    def close(self):
        if self.closed:return
        self.closed=True
        if self.engine.pack is self.wrapper:self.engine.pack=self.previous
        # Existing private modules keep numerically equivalent lazy CPU fallback
        # until the enclosing Owner closes its bank. Never reallocate unused
        # caches during cleanup or retain old functions merely to restore them.
        self.seen.clear()

def install(owner):
    owner.packed_cache_policy=PackedCachePolicy(owner)
    return owner.packed_cache_policy.close
