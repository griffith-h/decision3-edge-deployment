"""Fixed-shape whole text-attention graph; all unsupported calls retain the frozen path."""
from pathlib import Path
import numpy as np,torch
from graph import Graph

def install(model,graph_factory=Graph):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    layers=[(name,m) for name,m in model.backbone.named_modules() if type(m) is qwen.Qwen3_5Attention]
    if len(layers)!=6:raise ValueError('expected original six full-attention modules')
    originals=[];graphs=[];counts={'optimized':0,'fallback':0,'range_fallback':0,'disabled':0,'errors':[]}
    try:
        for i,(name,m) in enumerate(layers):
            g=graph_factory(Path('/work')/f'attention-{i}-fp16graph.rknn');graphs.append(g);original=m.forward;disabled=[False]
            def forward(hidden_states,position_embeddings,attention_mask,past_key_values=None,_g=g,_old=original,_m=m,_disabled=disabled,**kwargs):
                x=hidden_states;cos,sin=position_embeddings
                good=(not torch.is_grad_enabled() and not _m.training and not _disabled[0] and past_key_values is None and
                      x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.shape==(1,115,1024) and
                      all(v.device==x.device and v.dtype in (torch.float32,torch.bfloat16) and v.shape==(1,115,64) for v in (cos,sin)) and
                      attention_mask is not None and attention_mask.device==x.device and attention_mask.dtype==torch.bool and attention_mask.shape==(1,1,1,115) and
                      bool(attention_mask.any()) and all(k=='position_ids' or (k=='use_cache' and v is False) or (k=='output_attentions' and v is False) or v is None for k,v in kwargs.items()))
                def fallback():
                    counts['fallback']+=1
                    return _old(hidden_states,position_embeddings,attention_mask,past_key_values=past_key_values,**kwargs)
                if not good:return fallback()
                arrays={k:v.float().contiguous().numpy() for k,v in [('x',x),('cos',cos),('sin',sin)]}
                arrays['mask']=(~attention_mask).float().numpy()*-10000
                if any(not np.isfinite(v).all() or np.max(np.abs(v))>=65504 for v in arrays.values()):
                    counts['range_fallback']+=1;return fallback()
                try:out=_g.run(arrays)
                except RuntimeError as exc:
                    _disabled[0]=True;counts['disabled']+=1;counts['errors'].append(str(exc));return fallback()
                if not np.isfinite(out).all() or np.max(np.abs(out))>=65504:
                    counts['range_fallback']+=1;return fallback()
                counts['optimized']+=1
                return torch.from_numpy(out).bfloat16(),None
            m.forward=forward;originals.append((m,original))
    except Exception:
        for m,o in originals:m.forward=o
        for g in graphs:g.close()
        raise
    def restore():
        for m,o in originals:m.forward=o
        for g in graphs:g.close()
    return restore,counts
