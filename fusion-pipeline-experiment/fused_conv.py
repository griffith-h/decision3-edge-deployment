import ctypes,torch
from pathlib import Path
lib=ctypes.CDLL(str(Path(__file__).with_name('libfused_conv.so')));lib.d3_conv_fused.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*3;lib.d3_conv_fused.restype=ctypes.c_int
class FusedConv:
 def __init__(self,table):
  from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
  self.qwen=qwen;self.original=qwen.causal_conv1d_fn;self.table=table;self.enabled=False;self.closed=False;self.calls=0;self.disabled=False;self.errors=[];qwen.causal_conv1d_fn=self.run
 def run(self,x,weight,bias=None,activation=None,**kwargs):
  good=(self.enabled and not self.disabled and not torch.is_grad_enabled() and x.device.type=='cpu' and x.dtype==torch.bfloat16 and x.ndim==3 and 0<x.shape[0]<=8 and 512<x.shape[2]<=2048 and 0<x.shape[1]<=16384 and x.shape[1]%4==0 and weight.device==x.device and weight.dtype==x.dtype and weight.shape==(x.shape[1],4) and x.stride()==(x.shape[1]*x.shape[2],1,x.shape[1]) and activation in (None,'silu','swish') and all(v is None or (k=='use_cache' and v is False) for k,v in kwargs.items()) and (bias is None or (bias.device==x.device and bias.dtype==x.dtype and bias.shape==(x.shape[1],))))
  if not good:return self.original(x,weight,bias,activation=activation,**kwargs)
  b,c,t=x.shape;out=torch.empty((b,t,c),dtype=x.dtype);w=weight.contiguous();bb=bias.contiguous() if bias is not None else None
  rc=lib.d3_conv_fused(x.data_ptr(),w.data_ptr(),bb.data_ptr() if bb is not None else None,out.data_ptr(),self.table.silu.data_ptr() if activation else None,b,t,c)
  if rc:self.disabled=True;self.errors.append(rc);return self.original(x,weight,bias,activation=activation,**kwargs)
  self.calls+=1;return out.transpose(1,2)
 def close(self):
  if self.closed:return
  self.closed=True;self.qwen.causal_conv1d_fn=self.original
