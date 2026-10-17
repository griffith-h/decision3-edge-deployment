from baseline import load_baseline
from gpu_ops import install_attention
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
import torch,json
from pathlib import Path
class FakeGPU:
 def __init__(self):self.fail=False
 def attention(self,q,k,v,m):
  if self.fail:raise RuntimeError('injected GPU failure')
  return torch.zeros(q.shape[0],q.shape[2],q.shape[1],q.shape[3],dtype=q.dtype)
fake=FakeGPU();interface=qwen.ALL_ATTENTION_FUNCTIONS;orig=interface['sdpa'];calls=[];sentinel=(object(),object())
def spy(*args,**kwargs):calls.append((args,kwargs));return sentinel
interface.register('sdpa',spy);restore,c=install_attention(fake)
m=object.__new__(qwen.Qwen3_5Attention);torch.nn.Module.__init__(m);m.scaling=.5;m.eval()
q=torch.zeros(1,2,3,4,dtype=torch.bfloat16);k=q[:,:1].clone();v=k.clone();mask=torch.ones(1,1,1,3,dtype=torch.bool)
r={'cases':[]}
try:
 with torch.inference_mode():
  out,_=interface['sdpa'](m,q,k,v,attention_mask=mask);assert out.shape==(1,3,2,4)
  # Exact regression: a vision module passes attention_mask by keyword.
  vision=torch.nn.Identity().eval();assert interface['sdpa'](vision,q,k,v,attention_mask=mask) is sentinel;r['cases'].append('vision-keyword-mask')
  for tag,ms,kw in [('none',None,{}),('false',mask&False,{}),('causal',mask,{'is_causal':True}),('state',mask,{'use_cache':True}),('metadata',mask,{'unknown':'x'})]:
   assert interface['sdpa'](m,q,k,v,attention_mask=ms,**kw) is sentinel;r['cases'].append(tag)
  fake.fail=True;assert interface['sdpa'](m,q,k,v,attention_mask=mask) is sentinel;fake.fail=False;assert interface['sdpa'](m,q,k,v,attention_mask=mask) is sentinel;assert len(c['errors'])==1
 restore();assert interface['sdpa'] is spy;r['counts']=c;r['passed']=True
finally:interface.register('sdpa',orig)
Path('/work/logs/gpu-guards.json').write_text(json.dumps(r,indent=2)+'\n');print('GPU GUARDS PASS',flush=True)
