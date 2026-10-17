from baseline import load_baseline
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from cpu_attention import install
import torch,json
from pathlib import Path
r={'cases':[]};interface=qwen.ALL_ATTENTION_FUNCTIONS;orig=interface['sdpa'];calls=[];sentinel=(object(),object())
def spy(*args,**kwargs):calls.append((args,kwargs));return sentinel
interface.register('sdpa',spy)
m=object.__new__(qwen.Qwen3_5Attention);torch.nn.Module.__init__(m);m.scaling=.25;m.eval();restore,c=install()
fn=interface['sdpa'];q=torch.randn(1,2,3,16).bfloat16();k=torch.randn(1,1,3,16).bfloat16();v=torch.randn_like(k);mask=torch.ones(1,1,1,3,dtype=torch.bool)
try:
 with torch.inference_mode():
  out,_=fn(m,q,k,v,mask,scaling=.25);assert out.shape==(1,3,2,16) and c['optimized']==1
  for tag,ms,args,kwargs in [('none',None,(),{}),('all-false',mask&False,(),{}),('causal',mask,(),{'is_causal':True}),('dropout',mask,(),{'dropout':.2}),('cache',mask,(),{'use_cache':True}),('metadata',mask,(),{'unknown':'x'}),('positional',mask,(1,),{})]:
   assert fn(m,q,k,v,ms,*args,**kwargs) is sentinel;assert calls[-1][0][4] is ms and calls[-1][1]==kwargs;r['cases'].append(tag)
  bad=q.clone();bad[0,0,0,0]=float('nan');assert fn(m,bad,k,v,mask) is sentinel;assert c['range_fallback']==1
  m.train();assert fn(m,q,k,v,mask) is sentinel;m.eval()
 with torch.enable_grad():assert fn(m,q,k,v,mask) is sentinel
 restore();assert interface['sdpa'] is spy;r['passed']=True;r['counts']=c
finally:interface.register('sdpa',orig)
Path('/work/logs/cpu-attention-guards.json').write_text(json.dumps(r,indent=2)+'\n');print('ATTENTION GUARDS PASS',flush=True)
