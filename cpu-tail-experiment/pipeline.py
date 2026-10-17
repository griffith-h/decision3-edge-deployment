"""Keep original BF16 boundaries while eliminating MLP tensor round trips."""
import ctypes,torch,time
from types import MethodType
from lut_ops import good

def configure(engine):
 for name,args in [('mm_run_raw',[ctypes.c_void_p,ctypes.c_int]),('mm_prepare_mlp',[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p])]:
  f=getattr(engine.lib,name);f.argtypes=args;f.restype=ctypes.c_int


def run_pair(engine,hg,hu,rows):
 def raw(handle):engine.status(engine.lib.mm_run_raw(handle,rows))
 jobs=[engine.pool.submit(raw,h) for h in (hg,hu)];error=None
 for job in jobs:
  try:job.result()
  except Exception as e:
   if error is None:error=e
 if error is not None:raise error

def install(model,engine,table):
 configure(engine);restores=[];counts={'optimized':0,'fallback':0,'range_fallback':0,'raw_calls':0}
 for m in model.backbone.modules():
  if type(m).__name__!='Qwen3_5MLP':continue
  old=m.forward
  def forward(self,x,_old=old):
   members=[self.gate_proj,self.up_proj];down=self.down_proj
   if (self.training or not good(x) or x.ndim<2 or x.shape[-1]!=self.gate_proj.in_features or self.gate_proj.out_features!=self.up_proj.out_features or down.in_features!=self.gate_proj.out_features or engine.mode!='balanced' or
    not all(hasattr(m,'_bp_original') and m.bias is None and not m._forward_hooks and not m._forward_pre_hooks for m in members+[down]) or not all(engine.eligible(m,x) for m in members) or type(self.act_fn).__name__ not in ('SiLUActivation','SiLU')):
    counts['fallback']+=1;return _old(x)
   packed=engine.pack(members)
   pieces=members if packed is None else packed[0]
   if len(pieces)!=2 or any(m.out_features!=down.in_features for m in pieces):counts['fallback']+=1;return _old(x)
   g,u=pieces;hg,hu,hd=(engine.ensure(m) for m in (g,u,down));inp=x.contiguous();rows=x.numel()//x.shape[-1]
   engine.status(engine.lib.mm_prepare_bf16(hg,rows,inp.data_ptr()));engine.status(engine.lib.mm_copy_prepared(hu,hg,rows))
   run_pair(engine,hg,hu,rows)
   counts['raw_calls']+=2;status=engine.lib.mm_prepare_mlp(hd,hg,hu,rows,table.silu.data_ptr())
   if status in (-2,-3):counts['range_fallback']+=1;return _old(x)
   engine.status(status);out=torch.empty(*x.shape[:-1],down.out_features,dtype=x.dtype)
   status=engine.lib.mm_run_prepared_bf16(hd,rows,None,out.data_ptr())
   if status in (-2,-3):counts['range_fallback']+=1;return _old(x)
   engine.status(status);engine.calls+=3;counts['optimized']+=1;return out
  m.forward=MethodType(forward,m);restores.append(lambda m=m,old=old:setattr(m,'forward',old))
 def restore():
  for restore in reversed(restores):restore()
 return restore,counts
