from pathlib import Path
import numpy as np,torch
from core_graph import Graph

def install(model=None,graph_factory=Graph):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    interface=qwen.ALL_ATTENTION_FUNCTIONS;original=interface['sdpa'];g=graph_factory(Path('/work/attention-core.rknn'))
    counts={'optimized':0,'fallback':0,'range_fallback':0,'disabled':0,'errors':[]};disabled=[False]
    def forward(module,query,key,value,attention_mask,*args,**kwargs):
        good=(not torch.is_grad_enabled() and not module.training and not disabled[0] and type(module) is qwen.Qwen3_5Attention and not args and
              all(x.device.type=='cpu' and x.dtype==torch.bfloat16 for x in (query,key,value)) and query.shape==(1,8,115,256) and key.shape==value.shape==(1,2,115,256) and
              attention_mask is not None and attention_mask.dtype==torch.bool and attention_mask.device.type=='cpu' and attention_mask.shape==(1,1,1,115) and bool(attention_mask.any()) and
              not kwargs.get('use_cache',False) and kwargs.get('dropout',0)==0 and kwargs.get('scaling',.0625)==.0625 and not kwargs.get('is_causal',False) and
              all(k in ('dropout','scaling','position_ids','use_cache','is_causal') or v is None for k,v in kwargs.items()))
        def fallback():counts['fallback']+=1;return original(module,query,key,value,attention_mask,*args,**kwargs)
        if not good:return fallback()
        inputs={k:x.float().contiguous().numpy() for k,x in [('q',query),('k',key),('v',value)]};inputs['mask']=(~attention_mask).float().numpy()*-10000
        if any(not np.isfinite(x).all() or np.max(np.abs(x))>=65504 for x in inputs.values()):counts['range_fallback']+=1;return fallback()
        try:out=g.run(inputs)
        except RuntimeError as e:disabled[0]=True;counts['disabled']+=1;counts['errors'].append(str(e));return fallback()
        if not np.isfinite(out).all() or np.max(np.abs(out))>=65504:counts['range_fallback']+=1;return fallback()
        counts['optimized']+=1;return torch.from_numpy(out).bfloat16(),None
    interface.register('sdpa',forward)
    def restore():interface.register('sdpa',original);g.close()
    return restore,counts
