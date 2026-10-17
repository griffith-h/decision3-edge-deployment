"""Inductor CPU tensor islands; NPU calls remain outside compiled graphs."""
import time,os
from types import MethodType
import torch
import torch.nn.functional as F

def bf16_round_float(x):
    # Explicit bit rounding prevents CPU Inductor from bypassing BF16 casts.
    # Inference only; this is exactly the existing BF16 boundary, no new one.
    bits=x.contiguous().view(torch.int32)
    rounded=(bits+32767+((bits>>16)&1)) & -65536
    rounded=torch.where((bits & 0x7fffffff)>0x7f800000,torch.full_like(bits,0x7fc00000),rounded)
    return rounded.view(torch.float32)
def mlp_region(gate,up):return (bf16_round_float(F.silu(gate.float()))*up.float()).to(gate.dtype)
def norm_region(x,w,eps):
    z=x.float();return ((z*torch.rsqrt(z.square().mean(-1,keepdim=True)+eps))*(1+w.float())).to(x.dtype)
def gated_region(x,w,gate,eps):
    z=x.float();z=bf16_round_float(z*torch.rsqrt(z.square().mean(-1,keepdim=True)+eps))
    z=bf16_round_float(w.float()*z)
    return (z*F.silu(gate.float())).to(x.dtype)

def rope_region(q,k,cos,sin,unsqueeze_dim=1):
    cos=cos.unsqueeze(unsqueeze_dim);sin=sin.unsqueeze(unsqueeze_dim);dim=cos.shape[-1]
    def rotated(x):
        v=x[...,:dim];a,b=v.chunk(2,dim=-1);half=torch.cat((-b,a),dim=-1)
        left=v.float()*cos.float();right=half.float()*sin.float()
        low=(x.dtype==torch.bfloat16 and cos.dtype==torch.bfloat16 and sin.dtype==torch.bfloat16)
        if low:left=bf16_round_float(left);right=bf16_round_float(right)
        z=left+right
        if low:z=z.to(x.dtype)
        return torch.cat((z,x[...,dim:]),dim=-1)
    return rotated(q),rotated(k)

def install(model,control,stats):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    import torch._inductor.config as config
    config.emulate_precision_casts=True
    # Keep fusion's compiler OpenMP team within the same three-thread contract.
    config.cpp.threads=3
    config.compile_threads=1
    dynamic=os.environ.get('D3_COMPILE_DYNAMIC','1')!='0'
    compiled={name:torch.compile(fn,backend='inductor',fullgraph=True,dynamic=dynamic) for name,fn in
              [('mlp',mlp_region),('norm',norm_region),('gated',gated_region),('rope',rope_region)]}
    restores=[];disabled=set();seen=set();stats.update(calls={},first_signature_seconds={},errors=[],precision_cast_emulation=True,explicit_bf16_rounding=True,dynamic=dynamic,unseen_shape_fallbacks=0)
    def good(x):return not torch.is_grad_enabled() and x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.numel()>0
    def call(name,*args):
        signature=(name,tuple((tuple(v.shape),str(v.dtype),tuple(v.stride())) if isinstance(v,torch.Tensor) else v for v in args))
        if control.get('freeze_shapes') and signature not in seen:
            stats['unseen_shape_fallbacks']+=1;return None
        if name in disabled:return None
        t=time.perf_counter()
        try:out=compiled[name](*args)
        except Exception as exc:
            disabled.add(name);stats['errors'].append({'region':name,'error':repr(exc)});return None
        if signature not in seen:stats['first_signature_seconds'][str(signature)]=time.perf_counter()-t;seen.add(signature)
        stats['calls'][name]=stats['calls'].get(name,0)+1
        return out
    for name,module in model.backbone.named_modules():
        if 'language_model' not in name:continue
        original=module.forward
        if type(module) is qwen.Qwen3_5MLP:
            def mlp(self,x,_old=original):
                if control.get('mode') not in ('mlp','all') or not good(x):return _old(x)
                gate=self.gate_proj(x);up=self.up_proj(x)
                out=call('mlp',gate,up) if good(gate) and good(up) and gate.shape==up.shape else None
                return self.down_proj(self.act_fn(gate)*up if out is None else out)
            module.forward=MethodType(mlp,module)
        elif type(module) in (qwen.Qwen3_5RMSNorm,qwen.Qwen3_5RMSNormGated):
            kind=type(module) is qwen.Qwen3_5RMSNormGated;eps=module.variance_epsilon if kind else module.eps
            def norm(self,x,gate=None,_old=original,_kind=kind,_eps=eps):
                if control.get('mode')!='all' or not good(x) or not good(self.weight) or x.shape[-1]!=self.weight.numel() or (_kind and (gate is None or not good(gate) or gate.shape!=x.shape)):
                    return _old(x,gate) if _kind else _old(x)
                out=call('gated',x,self.weight,gate,_eps) if _kind else call('norm',x,self.weight,_eps)
                return (_old(x,gate) if _kind else _old(x)) if out is None else out
            module.forward=MethodType(norm,module)
        else:continue
        restores.append(lambda m=module,old=original:setattr(m,'forward',old))
    old_rope=qwen.apply_rotary_pos_emb
    def rope(q,k,cos,sin,unsqueeze_dim=1):
        if (control.get('mode')!='all' or not good(q) or not good(k) or cos.device.type!='cpu' or sin.device.type!='cpu'
            or cos.dtype!=sin.dtype or cos.dtype not in (torch.bfloat16,torch.float32) or cos.shape[-1]%2):
            return old_rope(q,k,cos,sin,unsqueeze_dim)
        out=call('rope',q,k,cos,sin,unsqueeze_dim)
        return old_rope(q,k,cos,sin,unsqueeze_dim) if out is None else out
    qwen.apply_rotary_pos_emb=rope;restores.append(lambda:setattr(qwen,'apply_rotary_pos_emb',old_rope))
    def restore():
        for cb in reversed(restores):cb()
    return restore
