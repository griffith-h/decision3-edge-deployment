"""Inference-only lookup fusion. Tables are process-owned immutable data."""
import ctypes,torch
from pathlib import Path
from types import MethodType
import torch.nn.functional as F
lib=ctypes.CDLL(str(Path(__file__).with_name('liblut_math.so')))
lib.d3_lut.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int];lib.d3_lut.restype=None
lib.d3_gate_table.argtypes=[ctypes.c_void_p];lib.d3_gate_table.restype=None
lib.d3_norm_lut.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_int];lib.d3_norm_lut.restype=None
class Tables:
 def __init__(self):
  with torch.inference_mode():
   values=torch.arange(65536,dtype=torch.int32).short().view(torch.bfloat16)
   self.silu=F.silu(values).contiguous();self.sigmoid=torch.sigmoid(values).contiguous();self.gate=torch.empty(65536,dtype=torch.float32)
   lib.d3_gate_table(self.gate.data_ptr())
def good(x):return isinstance(x,torch.Tensor) and not torch.is_grad_enabled() and x.device.type=='cpu' and x.dtype==torch.bfloat16 and 0<x.numel()<2**31
def activation(x,t,up=None,sigmoid=False):
 # Preserve dense permutations, including the real BTC-backed BCT convolution.
 dense=True;expected=1
 for stride,size in sorted(zip(x.stride(),x.shape)):
  if size<=1:continue
  if stride!=expected:dense=False;break
  expected*=size
 if up is None and dense:
  inp=x;out=torch.empty_strided(x.shape,x.stride(),dtype=x.dtype)
 else:
  inp=x.contiguous();up=up.contiguous() if up is not None else None;out=torch.empty_like(inp)
 lib.d3_lut(inp.data_ptr(),up.data_ptr() if up is not None else None,out.data_ptr(),(t.sigmoid if sigmoid else t.silu).data_ptr(),inp.numel())
 return out

def install(model,engine,t,variant):
 from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
 restores=[];counts={'activation':0,'mlp':0,'norm':0,'fallback':0}
 if 'activation' in variant:
  old_silu=F.silu;old_sigmoid=torch.sigmoid
  def silu(x,inplace=False):
   if not inplace and good(x):counts['activation']+=1;return activation(x,t)
   counts['fallback']+=1;return old_silu(x,inplace=inplace)
  def sigmoid(x,*,out=None):
   if out is None and good(x):counts['activation']+=1;return activation(x,t,sigmoid=True)
   counts['fallback']+=1;return old_sigmoid(x,out=out) if out is not None else old_sigmoid(x)
  F.silu=silu;torch.sigmoid=sigmoid
  restores.extend([lambda:setattr(F,'silu',old_silu),lambda:setattr(torch,'sigmoid',old_sigmoid)])
 for m in model.backbone.modules():
  if 'norm' in variant and type(m) in (qwen.Qwen3_5RMSNorm,qwen.Qwen3_5RMSNormGated):
   old=m.forward;kind=int(type(m) is qwen.Qwen3_5RMSNormGated);eps=m.variance_epsilon if kind else m.eps
   def norm(self,x,gate=None,_old=old,_kind=kind,_eps=eps):
    if (not good(x) or x.ndim==0 or x.shape[-1]==0 or self.weight.dtype!=x.dtype or self.weight.device!=x.device or x.shape[-1]!=self.weight.numel() or (_kind and (not good(gate) or gate.shape!=x.shape))):
     counts['fallback']+=1;return _old(x,gate) if _kind else _old(x)
    inp=x.contiguous();w=self.weight.contiguous();g=gate.contiguous() if _kind else None;out=torch.empty_like(inp)
    lib.d3_norm_lut(inp.data_ptr(),w.data_ptr(),g.data_ptr() if g is not None else None,out.data_ptr(),t.gate.data_ptr(),inp.numel()//inp.shape[-1],inp.shape[-1],_eps,_kind)
    counts['norm']+=1;return out
   m.forward=MethodType(norm,m);restores.append(lambda m=m,old=old:setattr(m,'forward',old))
  if 'mlp' in variant and type(m) is qwen.Qwen3_5MLP:
   old=m.forward
   def mlp(self,x,_old=old):
    if not good(x) or not all(hasattr(m,'_bp_original') and engine.eligible(m,x) for m in (self.gate_proj,self.up_proj)):
     counts['fallback']+=1;return _old(x)
    outputs=engine.group([self.gate_proj,self.up_proj],x);gate=outputs[id(self.gate_proj)];up=outputs[id(self.up_proj)]
    if not (good(gate) and good(up) and gate.shape==up.shape):raise RuntimeError('Unexpected projection output')
    counts['mlp']+=1;return self.down_proj(activation(gate,t,up=up))
   m.forward=MethodType(mlp,m);restores.append(lambda m=m,old=old:setattr(m,'forward',old))
 def restore():
  for r in reversed(restores):r()
 return restore,counts
