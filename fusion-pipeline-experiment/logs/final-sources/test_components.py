import sys
sys.path.insert(0,'/tail')
from runner import *
from pathlib import Path
import ctypes,threading
from npu_attention import NpuAttention,lib as npu_lib
from fused_conv import FusedConv
from lut_ops import Tables
import conv_btc
p=Path('/experiment');result={'attention':[],'conv':[],'passed':False};tables=Tables();torch.manual_seed(20261018)
with torch.inference_mode():
 native=NpuAttention(fused=True)
 try:
  q=torch.randn(1,8,615,256).bfloat16();k=torch.randn(1,2,615,256).bfloat16();v=torch.randn_like(k);mask=torch.ones(1,1,1,615,dtype=torch.bool)
  # Full future K/V access: when only the last key is visible, every query
  # equals that value, including the first query.
  last=torch.zeros_like(mask);last[:,:,:,-1]=True;out=native(q,k,v,last,1/16)
  for h in range(8):assert torch.equal(out[0,:,h],v[0,h//4,-1].expand(615,256))
  result['attention'].append('future-only key exactly reproduced')
  for which in ('q','k','v'):
   xs=[q.clone(),k.clone(),v.clone()];xs[('q','k','v').index(which)].reshape(-1)[-1]=float('nan')
   try:native(*xs,mask,1/16);raise AssertionError('accepted NaN')
   except RuntimeError:pass
  result['attention'].append('all Q/K/V inputs checked; NaN rejected')
  try:native(q,k,v,torch.zeros_like(mask),1/16);raise AssertionError('accepted all-masked')
  except RuntimeError:pass
  reference=native(q,k,v,mask,1/16);assert torch.equal(native(q,k,v,mask,1/16),reference);result['attention'].append('repeated call and recovery exact')
  original=npu_lib.dm_attention_worker;done=threading.Event();started=threading.Event()
  def injected(*args):
   first=args[-3]
   if first==0:
    started.wait(5);return -1
   started.set();value=original(*args);done.set();return value
  npu_lib.dm_attention_worker=injected
  try:
   try:native(q,k,v,mask,1/16);raise AssertionError('injection ignored')
   except RuntimeError:assert done.is_set(),'returned before other worker completed'
  finally:npu_lib.dm_attention_worker=original
  assert torch.equal(native(q,k,v,mask,1/16),reference);result['attention'].append('fault joins all workers before propagation; recovery exact')
 finally:native.close();native.close()
 assert not native.handles
 fused=FusedConv(tables);fused.enabled=True
 try:
  for b,t,c in ((1,513,128),(1,615,6144),(2,611,128),(1,1024,256)):
   x=torch.randn(b,t,c).bfloat16().transpose(1,2);w=torch.randn(c,4).bfloat16();bias=torch.randn(c).bfloat16()
   for has_bias in (False,True):
    bb=bias if has_bias else None;ref=torch.empty(b,t,c,dtype=x.dtype);assert conv_btc.fn(x.data_ptr(),w.data_ptr(),bb.data_ptr() if bb is not None else None,ref.data_ptr(),b,t,c)==0
    for activation in (None,'silu'):
     out=fused.run(x,w,bb,activation);expected=ref.transpose(1,2) if activation is None else torch.nn.functional.silu(ref.transpose(1,2));assert torch.equal(out,expected),(b,t,c,has_bias,activation,float((out.float()-expected.float()).abs().max()))
     result['conv'].append([b,t,c,has_bias,activation])
  saved=fused.original;sentinel=object();calls=[]
  fused.original=lambda *a,**kw:(calls.append(kw),sentinel)[1]
  assert fused.run(torch.zeros(1,4,512,dtype=torch.bfloat16),torch.zeros(4,4,dtype=torch.bfloat16)) is sentinel
  assert fused.run(x,w,bias,'silu',use_cache=True) is sentinel
  assert fused.run(x.float(),w.float()) is sentinel
  fused.original=saved;result['conv_fallbacks']=len(calls)
 finally:fused.close();fused.close()
result['passed']=True;(p/'logs/components-unit.json').write_text(json.dumps(result,indent=2)+'\n');print('COMPONENT UNIT PASS',result,flush=True)
