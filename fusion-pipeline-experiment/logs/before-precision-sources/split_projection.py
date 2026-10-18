"""Split selected single output projections across two independent core handles.

Exact output-column slices of unchanged weights. Existing group projections and
MLP graphs are untouched. Parent RowProjection owns whole-operation fallback.
"""
from concurrent.futures import wait
import torch
from safe_engine import OutputRangeError
class SplitProjection:
 def __init__(self,projections,engine,max_rows=512,min_rows=1):
  self.max_rows=max_rows;self.min_rows=min_rows
  self.engine=engine;self.original=engine.execute;self.enabled=False;self.closed=False;self.calls=0;self.splits={};self.selected={id(m):m for name,m in projections.modules.items() if name.endswith(('.o_proj','.out_proj')) and m.out_features%32==0 and m.out_features>=1024};self.states={key:(m.weight,m.weight._version,m.bias,m.bias._version if m.bias is not None else None) for key,m in self.selected.items()};engine.execute=self.execute
 def pieces(self,m):
  if id(m) in self.splits:return self.splits[id(m)]
  width=m.out_features//2;parts=[]
  with torch.inference_mode():
   for i in range(2):
    v=torch.nn.Linear.__new__(torch.nn.Linear);torch.nn.Module.__init__(v);v.in_features=m.in_features;v.out_features=width;v.weight=torch.nn.Parameter(m.weight[i*width:(i+1)*width].contiguous(),requires_grad=False);v.bias=torch.nn.Parameter(m.bias[i*width:(i+1)*width].float().contiguous(),requires_grad=False) if m.bias is not None else None;v._bp_handle=None
    self.engine.core[id(v)]=i+1;self.engine.bucket[id(v)]=i;parts.append(v)
  self.splits[id(m)]=parts;return parts
 def execute(self,module,x,prepared=False,output=None):
  if not self.enabled or id(module) not in self.selected or prepared or torch.is_grad_enabled() or x.dtype!=torch.bfloat16 or x.device.type!='cpu' or not self.min_rows<=x.numel()//x.shape[-1]<=self.max_rows:return self.original(module,x,prepared,output)
  w,ver,b,bver=self.states[id(module)]
  if module.weight is not w or w._version!=ver or module.bias is not b or (b is not None and b._version!=bver):return self.original(module,x,prepared,output)
  parts=self.pieces(module)
  for m in parts:self.engine.ensure(m)
  plans=self.engine.plan([module],parts);rows=x.numel()//x.shape[-1];inp=x.contiguous();hg,hu=(self.engine.ensure(m) for m in parts)
  self.engine.status(self.engine.lib.mm_prepare_bf16(hg,rows,inp.data_ptr()));self.engine.status(self.engine.lib.mm_copy_prepared(hu,hg,rows))
  out=self.engine.output_buffer(module,x) if output is None else output;jobs=[]
  try:
   for m,plan in zip(parts,plans):jobs.append(self.engine.pool.submit(self.engine.run_piece,m,x,{id(module):out},plan))
  finally:wait(jobs)
  error=None;range_failed=False
  for job in jobs:
   try:job.result()
   except OutputRangeError:range_failed=True
   except Exception as exc:
    if error is None:error=exc
  if error is not None:raise error
  if range_failed:raise ValueError('Split output outside FP16 range')
  self.calls+=1;return out
 def close(self):
  if self.closed:return
  self.closed=True;self.engine.execute=self.original;self.splits={}
