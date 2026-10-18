"""Only MLP rows are packed; sequence, masks and positions remain untouched."""
from pathlib import Path
from types import MethodType
import torch
from mlp_graph import HalfGraph,lib

class RowBank:
 def __init__(self,model,factory=HalfGraph,root=Path('/work'),max_rows=2048):
  self.mode='fixed';self.max_rows=max_rows;self.graphs=[];self.restores=[];self.disabled=set();self.counts={};self.closed=False;self.reset_counts()
  layers=[m for m in model.backbone.modules() if type(m).__name__=='Qwen3_5MLP'];assert len(layers)==12
  try:
   for i,m in enumerate(layers):
    g=factory(root/f'mlp-{i}-fp16graph.rknn');self.graphs.append(g)
    if factory is HalfGraph and lib.mlp_graph_set_core(g.handle,0):raise RuntimeError('Core selection failed')
    old=m.forward;weights=tuple(getattr(m,k).weight for k in ('gate_proj','up_proj','down_proj'));versions=tuple(w._version for w in weights);activation=m.act_fn
    def forward(layer,x,_i=i,_g=g,_old=old,_w=weights,_v=versions,_act=activation):
     signature=str(tuple(x.shape));self.counts['shapes'][signature]=self.counts['shapes'].get(signature,0)+1
     rows=x.shape[0]*x.shape[1] if x.ndim==3 else 0
     good=(self.mode!='cpu' and not torch.is_grad_enabled() and not layer.training and _i not in self.disabled and x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.ndim==3 and x.shape[-1]==1024 and 0<rows<=self.max_rows and layer.act_fn is _act and all(getattr(layer,k).weight is w and w._version==v and w.dtype==torch.bfloat16 and w.device.type=='cpu' and getattr(layer,k).bias is None for k,w,v in zip(('gate_proj','up_proj','down_proj'),_w,_v)) and not any(v._forward_hooks or v._forward_pre_hooks for v in (layer.gate_proj,layer.up_proj,layer.down_proj,layer.act_fn)))
     if self.mode=='fixed':good=good and tuple(x.shape)==(1,115,1024)
     def fallback():self.counts['fallback']+=1;return _old(x)
     if not good:return fallback()
     try:
      out=self.run_rows(_g,x)
     except ValueError:self.counts['range_fallback']+=1;return fallback()
     except RuntimeError as exc:self.disabled.add(_i);self.counts['errors'].append({'layer':_i,'error':str(exc)});return fallback()
     self.counts['optimized']+=1;return out
    m.forward=MethodType(forward,m);self.restores.append(lambda m=m,old=old:setattr(m,'forward',old))
  except BaseException:self.close();raise
 def reset_counts(self):self.counts={'optimized':0,'fallback':0,'range_fallback':0,'errors':[],'graph_calls':0,'padded_rows':0,'shapes':{}}
 def run_rows(self,g,x):
  return self.run_packed(g,x,115)
 def run_packed(self,g,x,width):
  flat=x.reshape(-1,1024);n=flat.shape[0]
  if n==width:
   self.counts['graph_calls']+=1;return g.run_bf16(flat.reshape(1,width,1024)).reshape(x.shape)
  result=torch.empty((n,1024),dtype=x.dtype,device=x.device)
  for start in range(0,n,width):
   used=min(width,n-start)
   if used==width:part=flat[start:start+width].reshape(1,width,1024)
   else:
    part=torch.zeros((1,width,1024),dtype=x.dtype,device=x.device);part[0,:used].copy_(flat[start:start+used]);self.counts['padded_rows']+=width-used
   self.counts['graph_calls']+=1;value=g.run_bf16(part);result[start:start+used].copy_(value[0,:used])
  return result.reshape(x.shape)
 def close(self):
  if self.closed:return
  self.closed=True
  for r in reversed(self.restores):r()
  for g in self.graphs:g.close()
