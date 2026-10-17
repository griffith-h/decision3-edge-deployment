import baseline,torch,numpy as np,json
from pathlib import Path
from attention_core import install
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
interface=qwen.ALL_ATTENTION_FUNCTIONS;before=interface['sdpa'];fallback=[]
def original(*args,**kw):fallback.append(kw);return 'original',None
interface.register('sdpa',original)
class Fake:
    instances=[]
    def __init__(self,path):self.calls=0;self.fail=False;self.closed=False;self.__class__.instances.append(self)
    def run(self,inputs):
        self.calls+=1
        if self.fail:raise RuntimeError('injected')
        return np.zeros((1,115,8,256),np.float32)
    def close(self):self.closed=True
restore,counts=install(None,Fake);fn=interface['sdpa'];m=qwen.Qwen3_5Attention.__new__(qwen.Qwen3_5Attention);torch.nn.Module.__init__(m);m.eval()
q=torch.zeros(1,8,115,256,dtype=torch.bfloat16);k=torch.zeros(1,2,115,256,dtype=torch.bfloat16);v=k.clone();mask=torch.ones(1,1,1,115,dtype=torch.bool)
try:
    with torch.inference_mode():
        out,_=fn(m,q,k,v,mask,scaling=.0625,dropout=0.,use_cache=False);assert out.shape==(1,115,8,256)
        for qq,mm,kw in [(q[:,:,:114],mask,{}),(q.float(),mask,{}),(q,torch.zeros_like(mask),{}),(q,mask,{'is_causal':True}),(q,mask,{'scaling':.125})]:assert fn(m,qq,k,v,mm,**kw)[0]=='original'
        Fake.instances[0].fail=True;assert fn(m,q,k,v,mask)[0]=='original';n=Fake.instances[0].calls;assert fn(m,q,k,v,mask)[0]=='original' and Fake.instances[0].calls==n
finally:restore();assert interface['sdpa'] is original;interface.register('sdpa',before)
assert Fake.instances[0].closed
Path('/work/logs/core-guard-unit.json').write_text(json.dumps({'passed':True,'counts':counts},indent=2)+'\n');print('PASS core guards',flush=True)
