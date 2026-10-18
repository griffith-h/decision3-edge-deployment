import sys
sys.path.insert(0,'/tail')
from runner import *
from npu_attention import NpuAttention
from pathlib import Path
p=Path('/experiment');r={'records':[],'passed':False};captured=torch.load('/tail/captured.pt',weights_only=False)['attention'];a=None;g=None
try:
 a=NpuAttention(fused=True);g=NpuAttention(fused=True,grouped=True)
 with torch.inference_mode():
  for i,row in enumerate(captured):
   q,k,v,mask,scale=row['args'];normal=a(q,k,v,mask,scale);out=g(q,k,v,mask,scale);delta=(out.float()-normal.float()).abs();r['records'].append({'layer':i,'max_hidden_delta':float(delta.max()),'mean_hidden_delta':float(delta.mean()),'allclose':torch.allclose(out.float(),normal.float(),atol=.002,rtol=.008)})
   assert r['records'][-1]['allclose']
   last=torch.zeros_like(mask);last[:,:,:,-1]=True;future=g(q,k,v,last,scale)
   for h in range(8):assert torch.equal(future[0,:,h],v[0,h//4,-1].expand(615,256))
  r['future_keys_exact']=True;r['passed']=True;(p/'logs/grouped-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('GROUPED UNIT PASS',r,flush=True)
except BaseException as exc:
 r['error']=repr(exc);raise
finally:
 if g:g.close()
 if a:a.close()
 r['handles_closed']=True;(p/'logs/grouped-unit.json').write_text(json.dumps(r,indent=2)+'\n')
