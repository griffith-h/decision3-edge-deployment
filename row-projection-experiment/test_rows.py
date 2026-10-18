from types import SimpleNamespace,MethodType
from pathlib import Path
import torch,json
from row_projection import RowProjection
class Engine:
 def __init__(self):self.cpu_fallbacks=0;self.groups=[{'parent':'language_model.layers.0.attn','members':['q_proj','k_proj','v_proj']}];self.calls=[];self.fail=None;self.kind='range'
 def execute(self,m,x):return m._bp_original(x)*2
 def group(self,ms,x):
  self.calls.append(x.shape[0]);assert x.shape[0]<=512
  if len(self.calls)==self.fail:
   if self.kind=='internal':self.cpu_fallbacks+=1;return {id(m):m._bp_original(x) for m in ms}
   raise (ValueError if self.kind=='range' else RuntimeError)('injected second block')
  return {id(m):self.execute(m,x) for m in ms}
def fixture(e):
 root=torch.nn.Module();root.language_model=torch.nn.Module();root.language_model.layers=torch.nn.ModuleList([torch.nn.Module()]);parent=torch.nn.Module();root.language_model.layers[0].attn=parent
 for i,(key,n) in enumerate(zip(['q_proj','k_proj','v_proj'],[8,16,24])):
  m=torch.nn.Linear(32,n,bias=True,dtype=torch.bfloat16);parent.add_module(key,m)
  with torch.no_grad():m.weight.zero_();m.weight[::2,0]=1;m.weight[1::2,1]=1;m.bias.fill_(i/8)
  m._bp_original=m.forward
  def old(layer,x):return e.execute(layer,x) if x.numel()//x.shape[-1]<=512 and x.dtype==torch.bfloat16 else layer._bp_original(x)
  m.forward=MethodType(old,m)
 def parent_forward(layer,x):return torch.cat([getattr(layer,k)(x) for k in ['q_proj','k_proj','v_proj']],dim=-1)
 parent.forward=MethodType(parent_forward,parent);root.eval();return SimpleNamespace(backbone=root),parent
e=Engine();model,parent=fixture(e);saved=[m.forward for m in [parent,parent.q_proj,parent.k_proj,parent.v_proj]];c=RowProjection(model,e);c.mode='chunked';records=[]
try:
 with torch.inference_mode():
  for shape in [(1,511,32),(1,512,32),(1,513,32),(1,1024,32),(1,1025,32),(1,2048,32),(2,257,32),(3,171,32),(513,32)]:
   x=torch.zeros(shape,dtype=torch.bfloat16);index=torch.arange(x.numel()//32).reshape(shape[:-1]);x[...,0]=(index//128).bfloat16()/32;x[...,1]=(index%128).bfloat16()/256;snap=x.clone();c.reset_counts();out=parent(x);expected=torch.cat([m._bp_original(x)*2 for m in [parent.q_proj,parent.k_proj,parent.v_proj]],-1);assert torch.equal(out,expected) and torch.equal(x,snap),shape;records.append(shape)
  x=torch.zeros(1,513,64,dtype=torch.bfloat16)[...,::2];index=torch.arange(513);x[0,:,0]=(index//128).bfloat16()/32;x[0,:,1]=(index%128).bfloat16()/256;assert not x.is_contiguous();assert torch.equal(parent(x),torch.cat([m._bp_original(x)*2 for m in [parent.q_proj,parent.k_proj,parent.v_proj]],-1))
  for kind in ['range','internal','runtime']:
   e.fail=len(e.calls)+2;e.kind=kind;c.reset_counts();out=parent(x);cpu=torch.cat([m._bp_original(x) for m in [parent.q_proj,parent.k_proj,parent.v_proj]],-1);assert torch.equal(out,cpu),(kind,c.counts)
   if kind=='runtime':
    before=len(e.calls);assert torch.equal(parent(x),cpu) and len(e.calls)==before
   else:assert c.counts['range_fallback']==1
   e.fail=None
finally:c.close();c.close()
assert [m.forward for m in [parent,parent.q_proj,parent.k_proj,parent.v_proj]]==saved
# Guard paths must keep the original full projection, including across batches.
e=Engine();model,parent=fixture(e);c=RowProjection(model,e);c.mode='chunked'
try:
 with torch.inference_mode():
  for x in [torch.zeros(1,2049,32,dtype=torch.bfloat16)]:
   assert torch.equal(parent(x),torch.cat([m._bp_original(x) for m in [parent.q_proj,parent.k_proj,parent.v_proj]],-1));assert not e.calls
  bad=torch.zeros(1,513,32,dtype=torch.float32)
  try:parent(bad);raise AssertionError('Expected original dtype error')
  except RuntimeError as exc:assert 'same dtype' in str(exc) and not e.calls
  hook=parent.q_proj.register_forward_pre_hook(lambda *a:None);x=torch.ones(1,513,32,dtype=torch.bfloat16);assert torch.equal(parent(x),torch.cat([m._bp_original(x) for m in [parent.q_proj,parent.k_proj,parent.v_proj]],-1)) and not e.calls;hook.remove()
  parent.train();assert torch.equal(parent(x),torch.cat([m._bp_original(x) for m in [parent.q_proj,parent.k_proj,parent.v_proj]],-1)) and not e.calls;parent.eval()
  parent.q_proj.weight.add_(1);x=torch.ones(1,513,32,dtype=torch.bfloat16);assert torch.equal(parent(x),torch.cat([m._bp_original(x) for m in [parent.q_proj,parent.k_proj,parent.v_proj]],-1));assert not e.calls
finally:c.close()
r={'passed':True,'shapes':records,'noncontiguous':True,'full_group_fallback_on_second_piece_failure':True,'internal_safe_engine_fallback_detected':True,'runtime_disables_members':True,'guards':True,'restoration':True};Path('/experiment/logs/unit.json').write_text(json.dumps(r,indent=2)+'\n');print('ROW PROJECTION UNIT PASS',flush=True)
