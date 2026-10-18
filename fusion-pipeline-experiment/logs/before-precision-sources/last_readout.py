"""Private last-pooling consumer optimization; public encoder remains unchanged.

All615 input tokens run through the first11 layers. The twelfth full-attention
layer computes K/V over all615 positions and the query/output required by D3's
last-position readout. Its pointwise residual/MLP/norm evaluate that same row.
No input, key, value, layer, mask, or GatedDeltaNet state is removed.
"""
from types import MethodType
import torch
import attention_prepare as ap
import residual
import tail_native as native
class LastReadout:
 def __init__(self,owner):
  self.owner=owner;self.model=owner.model;self.original=self.model.logits;self.enabled=False;self.disabled=False;self.closed=False;self.calls=0;self.bypasses=0;self.errors=[];self.last_metadata={};self.supported=False
  try:
   self.encoder=self.model.encoder;self.language=self.encoder.language_model;self.stack=list(self.encoder.stack);self.last=self.stack[-1][0];self.att=self.last.self_attn
   self.context=self.original.__func__.__globals__['sdpa_backends']
   self.supported=(type(self.encoder).__name__=='PromptEncoder' and len(self.stack)==12 and self.stack[-1][1] is True and self.last.block_type=='full_attention' and self.att.head_dim==256 and self.att.q_proj.out_features==4096 and self.att.k_proj.out_features==self.att.v_proj.out_features==512 and self.att.o_proj.in_features==2048 and self.att.o_proj.out_features==1024 and str(self.model.device)=='cpu' and self.model.attention_mode=='noncausal_full_attention' and self.model.readout_dtype=='float32' and not self.model.permutation_average and type(self.language.norm).__name__=='Qwen3_5RMSNorm')
   self.guarded_modules=list(self.last.modules())+[self.language.norm];self.methods=[(m,m.forward) for m in self.guarded_modules]
   self.parameters=[(m,name,x,x._version) for m in self.guarded_modules for name,x in m.named_parameters(recurse=False)];self.readout=self.model.readout;self.readout_version=self.readout._version
   self.lib=ap.lib
  except (AttributeError,KeyError,TypeError,RuntimeError):self.supported=False
  def logits(model,sequences,counts):return self.logits(sequences,counts)
  self.model.logits=MethodType(logits,self.model)
 def good(self,sequences,counts):
  if not self.supported or self.disabled or self.closed or len(sequences)!=1 or len(counts)!=1 or len(sequences[0])!=615:return False
  if torch.is_autocast_enabled('cpu') or self.model.readout is not self.readout or self.readout._version!=self.readout_version or self.model.readout_dtype!='float32' or self.model.permutation_average:return False
  if self.model.encoder is not self.encoder or self.model.backbone.training or self.encoder.language_model is not self.language or len(self.encoder.stack)!=12 or any(a is not b or x!=y for (a,x),(b,y) in zip(self.encoder.stack,self.stack)):return False
  if any(m.training or m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in self.model.backbone.modules()):return False
  live=list(self.last.modules())+[self.language.norm]
  if len(live)!=len(self.guarded_modules) or any(a is not b for a,b in zip(live,self.guarded_modules)):return False
  if any(m.forward!=fn for m,fn in self.methods) or any(getattr(m,name) is not x or x._version!=version or x.device.type!='cpu' for m,name,x,version in self.parameters):return False
  if not all(v.dtype==torch.bfloat16 for v in self.last.parameters()) or self.model.readout.dtype!=torch.float32:return False
  return True
 def q_project(self,x):return self.att.q_proj(x)
 def out_project(self,x):return self.att.o_proj(x)
 def mlp(self,x):return self.last.mlp(x)
 def attention(self,q,k,v,mask,scale):
  if not all(native.finite(x.contiguous()) for x in (q,k,v)):raise ValueError('Nonfinite last-query input')
  groups=q.shape[1]//k.shape[1];q=q.float();k=k.float().repeat_interleave(groups,dim=1);v=v.float().repeat_interleave(groups,dim=1)
  y=torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,is_causal=False,scale=scale).transpose(1,2).contiguous().bfloat16()
  if not native.finite(y):raise ValueError('Nonfinite last-query output')
  return y
 def last_layer(self,hidden,rope,mask):
  if tuple(hidden.shape)!=(1,615,1024) or hidden.dtype!=torch.bfloat16 or mask.dtype!=torch.bool or tuple(mask.shape)!=(1,1,1,615):raise ValueError('Unsupported final readout signature')
  cosine,sine=rope;R=cosine.shape[-1]
  if cosine.shape!=sine.shape or cosine.ndim!=3 or cosine.shape[:2]!=(1,615) or cosine.dtype!=torch.bfloat16 or not 0<R<=256 or R%8:raise ValueError('Unsupported final readout RoPE')
  residual_row=hidden[:,-1:].contiguous();x=self.last.input_layernorm(hidden)
  members=[self.att.k_proj,self.att.v_proj]
  if not self.owner.projections.good(members,x):raise ValueError('Final K/V projection guard')
  values=self.owner.projections.run(members,x,tuple(id(m) for m in members));kr=values[id(members[0])].contiguous();vr=values[id(members[1])]
  qr=self.q_project(x[:,-1:].contiguous())
  if not all(native.finite(z.contiguous()) for z in (qr,kr,vr)):raise ValueError('Nonfinite final projection result')
  qw=self.att.q_norm.weight.contiguous();kw=self.att.k_norm.weight.contiguous();q=torch.empty((1,8,1,256),dtype=hidden.dtype);k=torch.empty((1,2,615,256),dtype=hidden.dtype)
  c=cosine.contiguous();s=sine.contiguous();cl=c[:,-1:].contiguous();sl=s[:,-1:].contiguous()
  # Own frozen preparation kernel supports independently nonempty Q or K jobs:
  # with H=0 or KH=0, the other branch has zero jobs and is never dereferenced.
  self.lib.d3_attention_prepare(qr.data_ptr(),qr.data_ptr(),qw.data_ptr(),kw.data_ptr(),cl.data_ptr(),sl.data_ptr(),q.data_ptr(),k.data_ptr(),1,1,8,0,256,R,self.att.q_norm.eps,self.att.k_norm.eps)
  self.lib.d3_attention_prepare(qr.data_ptr(),kr.data_ptr(),qw.data_ptr(),kw.data_ptr(),c.data_ptr(),s.data_ptr(),q.data_ptr(),k.data_ptr(),1,615,0,2,256,R,self.att.q_norm.eps,self.att.k_norm.eps)
  v=vr.view(1,615,2,256).transpose(1,2);y=self.attention(q,k,v,mask,self.att.scaling);y=ap.gated(y,qr,self.owner.candidate.table);projected=self.out_project(y)
  residual_row,x=residual.add_norm(residual_row,projected,self.last.post_attention_layernorm);x=self.mlp(x);out,_=residual.add_norm(residual_row,x)
  self.last_metadata={'input_tokens':615,'kv_tokens':615,'query_rows':1,'prefix_layers':11,'final_layers':1,'total_layers':12,'scan_layers_unchanged':6}
  return out
 def encode_last(self,embeds,present):
  model=self.language;width=present.shape[1];ramp=torch.arange(width,device=embeds.device).view(1,1,-1).expand(4,1,-1);positions,mrope=ramp[0],ramp[1:];rope=model.rotary_emb(embeds,mrope);keys=present.bool()[:,None,None,:]
  scan=self.encoder.recurrent_mask(config=model.config,inputs_embeds=embeds,attention_mask=present,past_key_values=None,position_ids=positions)
  hidden=embeds
  for layer,softmax in self.stack[:-1]:hidden=layer(hidden,position_embeddings=rope,attention_mask=keys if softmax else scan,position_ids=positions,past_key_values=None,use_cache=False)
  return model.norm(self.last_layer(hidden,rope,keys))[:,-1]
 def logits(self,sequences,counts):
  if not self.enabled:return self.original(sequences,counts)
  if not self.good(sequences,counts):self.bypasses+=1;return self.original(sequences,counts)
  try:
   with torch.inference_mode(),self.context(self.model.device):
    ids=torch.as_tensor(sequences,dtype=torch.long,device=self.model.device);present=torch.ones_like(ids);embeds=self.model.backbone.get_input_embeddings()(ids);hidden=self.encode_last(embeds,present);logits=hidden.float()@self.model.readout.T
    limit=torch.as_tensor(list(counts),device=self.model.device)[:,None];invalid=torch.arange(255,device=self.model.device)[None]>=limit;out=logits.masked_fill(invalid,float('-inf'))
   self.calls+=1;return out
  except (ValueError,RuntimeError) as exc:self.errors.append(str(exc));self.disabled=True;return self.original(sequences,counts)
 def close(self):
  if self.closed:return
  self.closed=True;self.model.logits=self.original
