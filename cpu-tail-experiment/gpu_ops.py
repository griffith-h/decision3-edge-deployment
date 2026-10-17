import torch
from prepared_scan import prepare

def install_attention(gpu):
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 original=qwen.ALL_ATTENTION_FUNCTIONS['sdpa'];counts={'optimized':0,'fallback':0,'errors':[]};disabled=[False]
 def forward(module,q,k,v,attention_mask=None,*args,**kwargs):
  mask=attention_mask
  good=(not disabled[0] and not torch.is_grad_enabled() and not module.training and type(module) is qwen.Qwen3_5Attention and not args and
   all(x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.ndim==4 for x in (q,k,v)) and q.shape[0]==1 and 0<q.shape[1]<=16 and 0<q.shape[2]<=512 and 0<q.shape[3]<=256 and k.shape==v.shape and k.shape[0]==1 and k.shape[2:]==q.shape[2:] and k.shape[1]>0 and q.shape[1]%k.shape[1]==0 and
   mask is not None and mask.device.type=='cpu' and mask.dtype==torch.bool and mask.shape==(1,1,1,q.shape[2]) and bool(mask.any()) and
   kwargs.get('scaling',module.scaling)==q.shape[-1]**-.5 and not kwargs.get('is_causal',False) and not kwargs.get('use_cache',False) and kwargs.get('dropout',0)==0 and all(k in ('scaling','is_causal','use_cache','dropout','position_ids') or v is None for k,v in kwargs.items()))
  def fallback():counts['fallback']+=1;return original(module,q,k,v,mask,*args,**kwargs)
  if not good:return fallback()
  if not all(bool(torch.isfinite(x).all()) for x in (q,k,v)):return fallback()
  try:out=gpu.attention(q,k,v,mask)
  except (RuntimeError,OSError) as e:disabled[0]=True;counts['errors'].append(repr(e));return fallback()
  if not bool(torch.isfinite(out).all()):return fallback()
  counts['optimized']+=1;return out,None
 qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa',forward)
 def restore():qwen.ALL_ATTENTION_FUNCTIONS.register('sdpa',original)
 return restore,counts

def install_scan(gpu):
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 original=qwen.torch_chunk_gated_delta_rule;counts={'optimized':0,'fallback':0,'errors':[]};disabled=[False]
 def forward(query,key,value,g,beta,chunk_size=64,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
  good=(not disabled[0] and not torch.is_grad_enabled() and all(x.device.type=='cpu' for x in (query,key,value,g,beta)) and query.dtype==key.dtype==value.dtype==torch.bfloat16 and g.dtype in (torch.bfloat16,torch.float32) and beta.dtype in (torch.bfloat16,torch.float32) and
   query.ndim==key.ndim==value.ndim==4 and g.ndim==beta.ndim==3 and query.shape==key.shape and query.shape[:3]==value.shape[:3] and g.shape==beta.shape==query.shape[:3] and query.shape[0]==1 and 0<query.shape[1]<=512 and 0<query.shape[2]<=32 and 0<query.shape[3]<=128 and query.shape[3]%4==0 and 0<value.shape[-1]<=128 and value.shape[-1]%4==0 and
   all(x.stride(-1)==1 for x in (query,key,value)) and initial_state is None and not output_final_state and all(v is None or (k=='use_cache' and v is False) for k,v in kwargs.items()))
  def fallback():counts['fallback']+=1;return original(query,key,value,g,beta,chunk_size=chunk_size,initial_state=initial_state,output_final_state=output_final_state,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
  if not good:return fallback()
  q,k,v=prepare(query,key,value,use_qk_l2norm_in_kernel);b=beta.transpose(1,2).float().contiguous();decay=g.transpose(1,2).float().contiguous().exp()
  if not all(bool(torch.isfinite(x).all()) for x in (q,k,v,decay,b)):return fallback()
  try:out=gpu.scan(q,k,v,decay,b)
  except (RuntimeError,OSError) as e:disabled[0]=True;counts['errors'].append(repr(e));return fallback()
  if not bool(torch.isfinite(out).all()):return fallback()
  counts['optimized']+=1;return out,None
 qwen.torch_chunk_gated_delta_rule=forward
 def restore():qwen.torch_chunk_gated_delta_rule=original
 return restore,counts
