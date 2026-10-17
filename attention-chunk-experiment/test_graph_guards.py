import baseline,torch,numpy as np,json
from pathlib import Path
from attention_graph import install
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
class FakeGraph:
    created=[]
    def __init__(self,path):self.closed=False;self.calls=0;self.mode='ok';self.__class__.created.append(self)
    def run(self,values):
        self.calls+=1
        if self.mode=='fail':raise RuntimeError('injected graph failure')
        out=np.zeros((1,115,1024),dtype=np.float32)
        if self.mode=='nan':out.fill(np.nan)
        return out
    def close(self):self.closed=True
root=torch.nn.Module();fallbacks=[]
for i in range(6):
    m=qwen.Qwen3_5Attention.__new__(qwen.Qwen3_5Attention);torch.nn.Module.__init__(m);m.eval()
    def original(*args,**kw):fallbacks.append(kw);return 'original',None
    m.forward=original;root.add_module(str(i),m)
class Holder:backbone=root
r,counts=install(Holder(),FakeGraph);m=root[0] if hasattr(root,'__getitem__') else getattr(root,'0')
x=torch.randn(1,115,1024).bfloat16();cos=torch.ones(1,115,64).bfloat16();sin=torch.zeros_like(cos);mask=torch.ones(1,1,1,115,dtype=torch.bool);saved=x.clone()
with torch.inference_mode():
    out,_=m(x,(cos,sin),mask,use_cache=False);assert out.shape==x.shape and torch.equal(x,saved)
    for xx,mm,kw in [(x[:,:114],mask,{}),(x.float(),mask,{}),(x,torch.zeros_like(mask),{}),(x,mask,{'past_key_values':object()}),(x,mask,{'output_attentions':True})]:assert m(xx,(cos,sin),mm,**kw)[0]=='original'
    FakeGraph.created[0].mode='nan';assert m(x,(cos,sin),mask)[0]=='original';assert counts['range_fallback']==1
    FakeGraph.created[0].mode='fail';assert m(x,(cos,sin),mask)[0]=='original';n=FakeGraph.created[0].calls
    FakeGraph.created[0].mode='ok';assert m(x,(cos,sin),mask)[0]=='original';assert FakeGraph.created[0].calls==n
with torch.enable_grad():assert getattr(root,'1')(x,(cos,sin),mask)[0]=='original'
r();assert all(g.closed for g in FakeGraph.created)
# Partial initialization failure restores every forward and closes constructed contexts.
FakeGraph.created=[]
class FailCreate(FakeGraph):
    def __init__(self,path):
        if len(FakeGraph.created)==2:raise RuntimeError('injected create failure')
        super().__init__(path)
try:install(Holder(),FailCreate)
except RuntimeError:pass
else:raise AssertionError('create failure missing')
assert all(g.closed for g in FakeGraph.created) and len(FakeGraph.created)==2
Path('/work/logs/graph-guard-unit.json').write_text(json.dumps({'passed':True,'counts':counts,'partial_creation_cleanup':True},indent=2)+'\n');print('PASS graph guards',flush=True)
