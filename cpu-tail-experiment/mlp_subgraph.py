import torch,numpy as np
from pathlib import Path
from types import MethodType
from mlp_graph import Graph

def install(model,graph_factory=Graph):
 owners=[];restores=[];counts={'optimized':0,'fallback':0,'errors':[],'range_fallback':0};disabled=set()
 try:
  layers=[m for m in model.backbone.modules() if type(m).__name__=='Qwen3_5MLP'];assert len(layers)==12
  for i,m in enumerate(layers):
   g=graph_factory(Path('/work')/('mlp-'+str(i)+'-fp16graph.rknn'));owners.append(g);old=m.forward
   weights=tuple(getattr(m,k).weight for k in ('gate_proj','up_proj','down_proj'));versions=tuple(w._version for w in weights);activation=m.act_fn
   def forward(self,x,_g=g,_old=old,_i=i,_versions=versions,_weights=weights,_activation=activation):
    good=(not torch.is_grad_enabled() and not self.training and _i not in disabled and x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.shape==(1,115,1024) and self.act_fn is _activation and all(getattr(self,k).weight is w and w._version==v and w.dtype==torch.bfloat16 and w.device.type=='cpu' and getattr(self,k).bias is None for k,w,v in zip(('gate_proj','up_proj','down_proj'),_weights,_versions)) and not any(v._forward_hooks or v._forward_pre_hooks for v in (self.gate_proj,self.up_proj,self.down_proj,self.act_fn)))
    def fallback():counts['fallback']+=1;return _old(x)
    if not good:return fallback()
    inp=None
    if not hasattr(_g,'run_bf16'):
     inp=x.float().contiguous().numpy()
     if not np.isfinite(inp).all() or np.max(np.abs(inp))>=65504:counts['range_fallback']+=1;return fallback()
    try:
     out=_g.run_bf16(x) if hasattr(_g,'run_bf16') else _g.run(inp)
    except ValueError:counts['range_fallback']+=1;return fallback()
    except RuntimeError as e:disabled.add(_i);counts['errors'].append({'layer':_i,'error':str(e)});return fallback()
    if isinstance(out,torch.Tensor):
     # run_bf16 validates raw I/O ranges before its final BF16 rounding.
     counts['optimized']+=1;return out
    if not np.isfinite(out).all() or np.max(np.abs(out))>=65504:counts['range_fallback']+=1;return fallback()
    counts['optimized']+=1;return torch.from_numpy(out).bfloat16()
   m.forward=MethodType(forward,m);restores.append(lambda m=m,old=old:setattr(m,'forward',old))
 except BaseException:
  for r in reversed(restores):r()
  for g in owners:g.close()
  raise
 def restore():
  for r in reversed(restores):r()
  for g in owners:g.close()
 return restore,counts
