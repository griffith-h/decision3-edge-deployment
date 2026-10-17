import torch,ctypes
from types import MethodType
from lut_ops import good,lib
lib.d3_add_norm.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_float];lib.d3_add_norm.restype=None

def add_norm(x,y,norm=None):
 x=x.contiguous();y=y.contiguous();res=torch.empty_like(x);out=torch.empty_like(x) if norm is not None else None;w=norm.weight.contiguous() if norm is not None else None
 lib.d3_add_norm(x.data_ptr(),y.data_ptr(),res.data_ptr(),out.data_ptr() if out is not None else None,w.data_ptr() if w is not None else None,x.numel()//x.shape[-1],x.shape[-1],norm.eps if norm is not None else 0)
 return res,out

def install(model):
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 restores=[];counts={'optimized':0,'fallback':0,'boundary_fallback':0}
 for m in model.backbone.modules():
  if type(m) is not qwen.Qwen3_5DecoderLayer:continue
  old=m.forward
  def forward(self,hidden_states,position_embeddings,attention_mask=None,position_ids=None,past_key_values=None,_old=old,**kwargs):
   norm=self.post_attention_layernorm
   if (self.training or not good(hidden_states) or hidden_states.ndim<2 or hidden_states.shape[-1]!=self.hidden_size or past_key_values is not None or
    self.block_type not in ('linear_attention','full_attention') or type(norm) is not qwen.Qwen3_5RMSNorm or norm.weight.device!=hidden_states.device or norm.weight.dtype!=hidden_states.dtype or norm.weight.numel()!=self.hidden_size or norm._forward_hooks or norm._forward_pre_hooks):
    counts['fallback']+=1;return _old(hidden_states,position_embeddings,attention_mask,position_ids,past_key_values,**kwargs)
   residual=hidden_states;hidden_states=self.input_layernorm(hidden_states)
   if self.block_type=='linear_attention':hidden_states=self.linear_attn(hidden_states=hidden_states,cache_params=past_key_values,attention_mask=attention_mask,**kwargs)
   else:hidden_states,_=self.self_attn(hidden_states=hidden_states,attention_mask=attention_mask,position_ids=position_ids,past_key_values=past_key_values,position_embeddings=position_embeddings,**kwargs)
   if good(hidden_states) and hidden_states.shape==residual.shape:residual,hidden_states=add_norm(residual,hidden_states,norm)
   else:
    counts['boundary_fallback']+=1;residual=residual+hidden_states;hidden_states=norm(residual)
   hidden_states=self.mlp(hidden_states)
   if good(hidden_states) and hidden_states.shape==residual.shape:hidden_states,_=add_norm(residual,hidden_states)
   else:counts['boundary_fallback']+=1;hidden_states=residual+hidden_states
   counts['optimized']+=1;return hidden_states
  m.forward=MethodType(forward,m);restores.append(lambda m=m,old=old:setattr(m,'forward',old))
 def restore():
  for r in reversed(restores):r()
 return restore,counts
