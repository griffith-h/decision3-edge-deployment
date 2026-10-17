import ctypes,torch
from types import MethodType
from lut_ops import good,lib
lib.d3_attention_prepare.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*6+[ctypes.c_float]*2;lib.d3_attention_prepare.restype=None
lib.d3_attention_gate.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3;lib.d3_attention_gate.restype=None

def prepare(qraw,kraw,qw,kw,cos,sin,qe,ke,H,KH,D):
 B,T=qraw.shape[:2];R=cos.shape[-1];qo=torch.empty(B,H,T,D,dtype=qraw.dtype);ko=torch.empty(B,KH,T,D,dtype=kraw.dtype)
 tensors=[x.contiguous() for x in (qraw,kraw,qw,kw,cos,sin)]
 lib.d3_attention_prepare(*(x.data_ptr() for x in tensors),qo.data_ptr(),ko.data_ptr(),B,T,H,KH,D,R,qe,ke);return qo,ko

def gated(x,qraw,table):
 B,T,H,D=x.shape;x=x.contiguous();qraw=qraw.contiguous();out=torch.empty_like(x)
 lib.d3_attention_gate(x.data_ptr(),qraw.data_ptr(),out.data_ptr(),table.sigmoid.data_ptr(),B*T,H,D);return out.reshape(B,T,H*D)

def install(model,engine,table):
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 restores=[];counts={'optimized':0,'fallback':0,'boundary_fallback':0}
 for m in model.backbone.modules():
  if type(m) is not qwen.Qwen3_5Attention:continue
  old=m.forward
  def forward(self,hidden_states,position_embeddings,attention_mask,past_key_values=None,_old=old,**kwargs):
   D=self.head_dim;H=self.q_proj.out_features//(2*D);KH=self.k_proj.out_features//D;cos,sin=position_embeddings
   members=[self.q_proj,self.k_proj,self.v_proj]
   good_input=(not self.training and good(hidden_states) and hidden_states.ndim==3 and hidden_states.shape[0]==1 and past_key_values is None and D%4==0 and all(good(x) for x in (cos,sin,self.q_norm.weight,self.k_norm.weight)) and cos.shape==sin.shape and cos.ndim==3 and cos.shape[:2]==hidden_states.shape[:2] and 0<cos.shape[-1]<=D and cos.shape[-1]%8==0 and
    all(hasattr(m,'_bp_original') and engine.eligible(m,hidden_states) for m in members) and not self.q_norm._forward_hooks and not self.k_norm._forward_hooks and not self.q_norm._forward_pre_hooks and not self.k_norm._forward_pre_hooks and self.config._attn_implementation=='sdpa' and kwargs.get('dropout',0)==0 and not kwargs.get('is_causal',False) and not kwargs.get('use_cache',False) and all(k in ('position_ids','dropout','is_causal','use_cache') or v is None for k,v in kwargs.items()))
   if not good_input:counts['fallback']+=1;return _old(hidden_states,position_embeddings,attention_mask,past_key_values,**kwargs)
   outputs=engine.group(members,hidden_states);qr=outputs[id(self.q_proj)];kr=outputs[id(self.k_proj)];vr=outputs[id(self.v_proj)]
   if not all(good(x) for x in (qr,kr,vr)):counts['boundary_fallback']+=1;return _old(hidden_states,position_embeddings,attention_mask,past_key_values,**kwargs)
   q,k=prepare(qr,kr,self.q_norm.weight,self.k_norm.weight,cos,sin,self.q_norm.eps,self.k_norm.eps,H,KH,D)
   B,T=hidden_states.shape[:2];v=vr.view(B,T,KH,D).transpose(1,2)
   forward_kwargs={k:v for k,v in kwargs.items() if k!='dropout'}
   out,weights=qwen.ALL_ATTENTION_FUNCTIONS['sdpa'](self,q,k,v,attention_mask,dropout=0.,scaling=self.scaling,**forward_kwargs)
   out=gated(out,qr,table);counts['optimized']+=1;return self.o_proj(out),weights
  m.forward=MethodType(forward,m);restores.append(lambda m=m,old=old:setattr(m,'forward',old))
 def restore():
  for restore in reversed(restores):restore()
 return restore,counts
