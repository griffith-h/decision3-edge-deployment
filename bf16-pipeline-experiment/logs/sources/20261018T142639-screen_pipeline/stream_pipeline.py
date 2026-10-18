"""Private zero-state linear decoder pipeline, using frozen model preparation.

C publishes complete 32-row groups for all heads. One consumer owns all SDK
projection/MLP contexts; producer only touches disjoint scan output rows. The
complete result stays private until producer and consumer have both joined.
"""
import ctypes,queue,threading
from types import MethodType
from pathlib import Path
import torch
import tail_native as native
import long_kernels as lk
import residual
READY=ctypes.CFUNCTYPE(ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p)
class StreamPipeline:
 def __init__(self,owner):
  self.owner=owner;self.enabled=False;self.parallel=True;self.group=128;self.threads=3;self.closed=False;self.calls=0;self.chunks=0;self.errors=[];self.disabled=set();self.restores=[];self.active=None
  self.lib=ctypes.CDLL(str(Path(__file__).with_name('libstream_scan.so')))
  for name,fn in [('d3_set_sgemm',lk.blas.scipy_cblas_sgemm64_),('d3_set_strsm',lk.blas.scipy_cblas_strsm64_)]:
   getattr(self.lib,name).argtypes=[ctypes.c_void_p];getattr(self.lib,name)(ctypes.cast(fn,ctypes.c_void_p))
  self.lib.d3_stream_scan.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*8+[READY,ctypes.c_void_p];self.lib.d3_stream_scan.restype=ctypes.c_int
  try:
   for layer,full in owner.model.encoder.stack:
    if full:continue
    old=layer.forward;modules=list(layer.modules());methods=[(m,m.forward) for m in modules if m is not layer];weights=[(m,n,w,w._version) for m in modules for n,w in m.named_parameters(recurse=False)];cpu=[(m,m._bp_original) for m in modules if hasattr(m,'_bp_original')]
    def forward(block,hidden_states,position_embeddings,attention_mask=None,position_ids=None,past_key_values=None,_old=old,_modules=modules,_methods=methods,_weights=weights,_cpu=cpu,**kwargs):
     good=(self.enabled and not self.closed and id(block) not in self.disabled and self.owner.float_active and not torch.is_grad_enabled() and not torch.is_autocast_enabled('cpu') and hidden_states.device.type=='cpu' and hidden_states.dtype==torch.bfloat16 and tuple(hidden_states.shape)==(1,615,1024) and attention_mask is None and past_key_values is None and all(k=='use_cache' and v is False for k,v in kwargs.items()) and block.block_type=='linear_attention' and all(not m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks for m in _modules) and list(block.modules())==_modules and all(m.forward==fn for m,fn in _methods) and all(getattr(m,n) is w and w._version==v and w.device.type=='cpu' for m,n,w,v in _weights) and all(m._bp_original==fn for m,fn in _cpu))
     if not good:return _old(hidden_states,position_embeddings,attention_mask=attention_mask,position_ids=position_ids,past_key_values=past_key_values,**kwargs)
     try:out=self.run(block,hidden_states)
     except (ValueError,RuntimeError) as exc:
      self.errors.append(str(exc));self.disabled.add(id(block));return _old(hidden_states,position_embeddings,attention_mask=attention_mask,position_ids=position_ids,past_key_values=past_key_values,**kwargs)
     self.calls+=1;return out
    layer.forward=MethodType(forward,layer);self.restores.append((layer,old))
  except BaseException:self.close();raise
 def run(self,block,hidden):
  m=block.linear_attn;x=block.input_layernorm(hidden)
  # Same zero-state public preparation as the frozen Qwen3_5GatedDeltaNet.
  if (m.num_v_heads,m.head_v_dim,m.head_k_dim,m.num_k_heads)!=(16,128,128,16):raise ValueError('Unsupported scan head structure')
  members=[m.in_proj_qkv,m.in_proj_z]
  if not self.owner.projections.good(members,x):raise ValueError('Input projection guard')
  values=self.owner.projections.run(members,x,tuple(id(v) for v in members));mixed=values[id(m.in_proj_qkv)].transpose(1,2);z=values[id(m.in_proj_z)].reshape(1,615,-1,m.head_v_dim)
  b=m.in_proj_b(x);a=m.in_proj_a(x)
  qwen=self.owner.long.qwen
  mixed=qwen.causal_conv1d_fn(mixed,m.conv1d.weight.squeeze(1),m.conv1d.bias,activation=m.activation,use_cache=False).transpose(1,2)
  query,key,value=torch.split(mixed,[m.key_dim,m.key_dim,m.value_dim],dim=-1)
  query=query.reshape(1,615,-1,m.head_k_dim);key=key.reshape(1,615,-1,m.head_k_dim);value=value.reshape(1,615,-1,m.head_v_dim)
  beta=b.sigmoid();g=-m.A_log.float().exp()*torch.nn.functional.softplus(a.float()+m.dt_bias)
  if m.num_v_heads//m.num_k_heads>1:
   query=query.repeat_interleave(m.num_v_heads//m.num_k_heads,dim=2);key=key.repeat_interleave(m.num_v_heads//m.num_k_heads,dim=2)
  if not native.validate_raw(query,key,value,g,beta):raise ValueError('Scan input domain')
  from prepared_scan import prepare
  q,k,v=prepare(query,key,value,True);decay=g.transpose(1,2).float().contiguous().exp();beta=beta.transpose(1,2).float().contiguous()
  if not all(native.finite(t) for t in (q,k,v,decay)):raise ValueError('Prepared scan domain')
  out=torch.empty(v.shape,dtype=torch.bfloat16);result=torch.empty_like(hidden);items=queue.Queue();errors=[]
  def consume(start,rows):
   part=out[:,:,start:start+rows].transpose(1,2).contiguous().reshape(-1,m.head_v_dim);zz=z[:,start:start+rows].reshape(-1,m.head_v_dim)
   normalized=m.norm(part,zz).reshape(1,rows,-1);projected=self.owner.engine.execute(m.out_proj,normalized)
   if not native.finite(projected.contiguous()):raise ValueError('Projection output domain')
   rr,xx=residual.add_norm(hidden[:,start:start+rows].contiguous(),projected,block.post_attention_layernorm)
   mlp=self.owner.precise_mlp.run((block.mlp.gate_proj,block.mlp.up_proj,block.mlp.down_proj),xx)
   yy,_=residual.add_norm(rr,mlp);result[:,start:start+rows].copy_(yy);self.chunks+=1
  def worker():
   with torch.inference_mode():
    while True:
     item=items.get()
     if item is None:return
     if errors:continue
     try:consume(*item)
     except BaseException as exc:errors.append(exc)
  @READY
  def ready(start,rows,user):
   try:
    if self.parallel:items.put((start,rows))
    else:consume(start,rows)
    return 0
   except BaseException as exc:errors.append(exc);return 1
  thread=threading.Thread(target=worker,name='d3-scan-consumer') if self.parallel else None
  self.active=thread
  try:
   if thread:thread.start()
   rc=self.lib.d3_stream_scan(*(t.data_ptr() for t in (q,k,v,decay,beta,out)),q.shape[0]*q.shape[1],615,q.shape[3],v.shape[3],32,64,self.group,self.threads,ready,None)
  finally:
   if thread:items.put(None);thread.join()
   self.active=None
  if errors:
   exc=errors[0]
   if isinstance(exc,(ValueError,RuntimeError)):raise exc
   raise RuntimeError('Consumer failed: '+repr(exc)) from exc
  if rc:raise RuntimeError('Stream scan failed: '+str(rc))
  if not native.finite(result):raise ValueError('Layer output domain')
  return result
 def close(self):
  if self.closed:return
  self.closed=True
  if self.active:self.active.join()
  for layer,old in reversed(self.restores):layer.forward=old
  self.restores=[]
