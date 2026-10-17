from baseline import load_baseline,stats
from lut_ops import Tables,activation
import torch,time,json
import torch.nn.functional as F
from pathlib import Path
model,engine,c,close=load_baseline();tab=Tables();old=F.silu;olds=torch.sigmoid;captures=[];body=json.loads(Path('/base/request.json').read_text())
def si(x,inplace=False):
 out=old(x,inplace=inplace)
 if x.dtype==torch.bfloat16:captures.append((x.detach().clone(memory_format=torch.preserve_format),out.detach().clone(memory_format=torch.preserve_format),False))
 return out
def sg(x,*,out=None):
 res=olds(x,out=out) if out is not None else olds(x)
 if out is None and x.dtype==torch.bfloat16:captures.append((x.detach().clone(memory_format=torch.preserve_format),res.detach().clone(memory_format=torch.preserve_format),True))
 return res
r={'calls':[]}
try:
 model.system_one(state=body['state'],questions=body['questions']);F.silu=si;torch.sigmoid=sg
 model.system_one(state=body['state'],questions=body['questions']);F.silu=old;torch.sigmoid=olds
 with torch.inference_mode():
  for x,ref,sig in captures:
   a=activation(x,tab,sigmoid=sig);bits=(a.view(torch.int16)!=ref.view(torch.int16));row={'shape':list(x.shape),'strides':list(x.stride()),'out_strides':list(a.stride()),'reference_strides':list(ref.stride()),'sigmoid':sig,'different':bits.sum().item(),'max_delta':(a.float()-ref.float()).abs().max().item()}
   for name,fn in [('stock',lambda:(olds(x) if sig else old(x))),('lut',lambda:activation(x,tab,sigmoid=sig))]:
    vals=[]
    for _ in range(12):t=time.perf_counter();fn();vals.append(time.perf_counter()-t)
    row[name]=stats(vals)
   r['calls'].append(row)
 r['passed']=all(v['different']==0 for v in r['calls']);Path('/work/logs/activation-replay.json').write_text(json.dumps(r,indent=2)+'\n');print('ACTUAL LUT EXACT',r['passed'],len(captures),flush=True)
finally:F.silu=old;torch.sigmoid=olds;close()
