from projected_owner import *
from scattered_adapter import PackedEngine
from types import SimpleNamespace,MethodType
from row_projection import RowProjection
from pathlib import Path
records=[]
for sizes in [(512,256,256),(768,256)]:
 e=PackedEngine('balanced');e.groups=[{'parent':'language_model.layers.0.attn','members':['p'+str(i) for i in range(len(sizes))]}]
 root=torch.nn.Module();root.language_model=torch.nn.Module();root.language_model.layers=torch.nn.ModuleList([torch.nn.Module()]);parent=torch.nn.Module();root.language_model.layers[0].attn=parent
 for i,n in enumerate(sizes):
  m=torch.nn.Linear(1024,n,bias=True,dtype=torch.bfloat16)
  with torch.no_grad():m.weight.zero_();m.weight[:,0]=((torch.arange(n)%4+1).float()/16).bfloat16();m.bias.fill_(i/8)
  m._bp_original=m.forward;m._bp_handle=None;e.core[id(m)]=1+(i>0);e.bucket[id(m)]=int(i>0)
  def old(layer,x):return e.execute(layer,x) if e.eligible(layer,x) else layer._bp_original(x)
  m.forward=MethodType(old,m);parent.add_module('p'+str(i),m)
 def pf(layer,x):return torch.cat([getattr(layer,'p'+str(i))(x) for i in range(len(sizes))],-1)
 parent.forward=MethodType(pf,parent);root.eval();c=RowProjection(SimpleNamespace(backbone=root),e)
 try:
  with torch.inference_mode():
   for mode in ['single','chunked']:
    c.mode=mode
    for B,T in [(1,513),(1,1024),(1,1025),(2,257),(1,2048)]:
     x=torch.zeros(B,T,2048,dtype=torch.bfloat16)[...,::2];x[...,0]=((torch.arange(B*T).reshape(B,T)%4+1).float()/8).bfloat16();assert not x.is_contiguous();c.reset_counts();out=parent(x);ref=torch.cat([getattr(parent,'p'+str(i))._bp_original(x) for i in range(len(sizes))],-1);assert torch.equal(out,ref),(mode,B,T,(out.float()-ref.float()).abs().max());assert c.counts['optimized_projections']==len(sizes) and c.counts['native_max_rows']<=512 and not c.counts['range_fallback'];records.append({'sizes':sizes,'mode':mode,'B':B,'T':T,'exact':True,'counts':c.counts})
  print('NATIVE GROUP PASS',sizes,flush=True)
 finally:c.close();e.close()
(P/'logs/native-unit.json').write_text(json.dumps({'passed':True,'records':records},indent=2)+'\n')
