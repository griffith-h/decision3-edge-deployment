"""Scoped inference-only SIMD math. Original BF16 rounding boundaries retained."""
import ctypes
from pathlib import Path
from types import MethodType
import torch
import torch.nn.functional as F

lib=ctypes.CDLL(str(Path(__file__).with_name('libcpu_math.so')))
lib.d3_activation.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2
lib.d3_norm.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_int]
def activation(x,up=None,kind=0):
    assert x.dtype==torch.bfloat16 and x.device.type=='cpu'
    x=x.contiguous();up=up.contiguous() if up is not None else None
    out=torch.empty_like(x)
    lib.d3_activation(x.data_ptr(),up.data_ptr() if up is not None else None,out.data_ptr(),x.numel(),kind)
    return out
def install(model,control):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    restores=[]
    original_silu=F.silu;original_sigmoid=torch.sigmoid
    def good(x):return not torch.is_grad_enabled() and x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.numel()<2**31
    def silu(x,inplace=False):
        if control.get('activation') and not inplace and good(x):return activation(x)
        return original_silu(x,inplace=inplace)
    def sigmoid(x,*,out=None):
        if control.get('activation') and out is None and good(x):return activation(x,kind=2)
        return original_sigmoid(x,out=out) if out is not None else original_sigmoid(x)
    F.silu=silu;torch.sigmoid=sigmoid
    restores.extend([lambda:setattr(F,'silu',original_silu),lambda:setattr(torch,'sigmoid',original_sigmoid)])
    for module in model.backbone.modules():
        if type(module) in (qwen.Qwen3_5RMSNorm,qwen.Qwen3_5RMSNormGated):
            original=module.forward;kind=int(type(module) is qwen.Qwen3_5RMSNormGated)
            eps=module.variance_epsilon if kind else module.eps
            def norm(self,x,gate=None,_old=original,_kind=kind,_eps=eps):
                if (not control.get('norm') or not good(x) or self.weight.dtype!=x.dtype or self.weight.device!=x.device or
                    x.shape[-1]!=self.weight.numel() or (_kind and (gate is None or gate.shape!=x.shape or gate.dtype!=x.dtype or gate.device!=x.device))):
                    return _old(x,gate) if _kind else _old(x)
                inp=x.contiguous();w=self.weight.contiguous();g=gate.contiguous() if _kind else None
                out=torch.empty_like(inp)
                lib.d3_norm(inp.data_ptr(),w.data_ptr(),g.data_ptr() if g is not None else None,out.data_ptr(),
                            inp.numel()//inp.shape[-1],inp.shape[-1],_eps,_kind)
                return out
            module.forward=MethodType(norm,module)
            restores.append(lambda m=module,old=original:setattr(m,'forward',old))
        if type(module) is qwen.Qwen3_5MLP:
            original=module.forward
            def mlp(self,x,_old=original):
                if not control.get('mlp') or not good(x):return _old(x)
                gate=self.gate_proj(x);up=self.up_proj(x)
                if good(gate) and good(up) and gate.shape==up.shape:
                    return self.down_proj(activation(gate,up,kind=1))
                return self.down_proj(self.act_fn(gate)*up)
            # Install before the unchanged bridge adapter so its parallel scope
            # still wraps this dispatcher and both independent projections.
            module.forward=MethodType(mlp,module)
            restores.append(lambda m=module,old=original:setattr(m,'forward',old))
    def restore():
        for callback in reversed(restores):callback()
    return restore
