from baseline import stats
from gpu_client import GPU
from prepared_scan import prepare
from chunk_blas2 import lib as scanlib
import torch,json,time
from pathlib import Path
r={'attention':[],'scan':[],'replay':[]};torch.set_num_threads(3);torch.manual_seed(19);gpu=GPU()
try:
 with torch.inference_mode():
  for t in (1,3,17,115,129,512):
   for maskkind in ('all','left','last'):
    h,kh,d=8,2,256 if t==115 else 32
    q=torch.randn(1,h,t,d).bfloat16();k=torch.randn(1,kh,t,d).bfloat16();v=torch.randn_like(k);m=torch.ones(1,1,1,t,dtype=torch.bool)
    if maskkind=='left':m[...,:t//2]=False
    if maskkind=='last':m[...,:-1]=False
    saved=[x.clone() for x in (q,k,v,m)]
    ref=torch.nn.functional.scaled_dot_product_attention(q.float(),k.float().repeat_interleave(h//kh,1),v.float().repeat_interleave(h//kh,1),m,scale=d**-.5).transpose(1,2).bfloat16()
    out=gpu.attention(q,k,v,m);torch.testing.assert_close(out,ref,atol=.008,rtol=.008)
    if maskkind=='last':assert torch.equal(out,ref)
    assert all(torch.equal(x,y) for x,y in zip((q,k,v,m),saved));r['attention'].append({'t':t,'mask':maskkind,'delta':(out.float()-ref.float()).abs().max().item()})
  for t in (1,3,17,115,129,512):
   h,kd,vd=2,32,16;q=torch.randn(1,h,t,kd)*.1;k=torch.randn_like(q)*.1;v=torch.randn(1,h,t,vd);decay=torch.rand(1,h,t);beta=torch.rand_like(decay)
   if t in (3,115):decay[:,:,0]=0;decay[:,:,1]=1e-30;beta[:,:,-1]=0
   state=torch.zeros(1,h,kd,vd,dtype=torch.float64);out=[]
   for i in range(t):
    state*=decay[:,:,i,None,None].double();pred=(state*k[:,:,i,:,None].double()).sum(-2);delta=(v[:,:,i].double()-pred)*beta[:,:,i,None].double();state+=k[:,:,i,:,None].double()*delta[:,:,:,None].transpose(-1,-2);out.append((state*q[:,:,i,:,None].double()).sum(-2))
   ref=torch.stack(out,1).bfloat16();actual=gpu.scan(q,k,v,decay,beta);torch.testing.assert_close(actual,ref,atol=.008,rtol=.008)
   if t==115:
    prefix=gpu.scan(q[:,:,:17],k[:,:,:17],v[:,:,:17],decay[:,:,:17],beta[:,:,:17]);assert torch.equal(prefix,actual[:,:17]);assert torch.equal(gpu.scan(q,k,v,decay,beta),actual)
   r['scan'].append({'t':t,'delta':(actual.float()-ref.float()).abs().max().item()})
  captures=torch.load('/previous/profile-candidate-inputs.pt',weights_only=False)
  for idx,row in enumerate(captures):
   q0,k0,v0,g,beta=row['tensors'];q,k,v=prepare(q0,k0,v0,row['kwargs'].get('use_qk_l2norm_in_kernel',False));b=beta.transpose(1,2).float().contiguous();decay=g.transpose(1,2).float().contiguous().exp();ref=torch.empty_like(v,dtype=torch.bfloat16)
   assert scanlib.d3_chunk_scan(*(x.data_ptr() for x in (q,k,v,decay,b,ref)),q.shape[0]*q.shape[1],q.shape[2],q.shape[3],v.shape[-1],32,64)==0
   expected=ref.transpose(1,2).contiguous();actual=gpu.scan(q,k,v,decay,b);torch.testing.assert_close(actual,expected,atol=.008,rtol=.008)
   times=[]
   for _ in range(12):start=time.perf_counter();actual=gpu.scan(q,k,v,decay,b);times.append(time.perf_counter()-start)
   r['replay'].append({'layer':idx,'delta':(actual.float()-expected.float()).abs().max().item(),'timing':stats(times)})
   print('GPU SCAN REPLAY',idx,r['replay'][-1]['timing']['median'],flush=True)
  r['passed']=True
finally:
 gpu.close();Path('/work/logs/gpu-unit.json').write_text(json.dumps(r,indent=2)+'\n')
print('GPU UNIT PASS',flush=True)
