import torch,json,glob,ctypes,statistics,time
from pathlib import Path
from cpu_attention import attention,blas
from baseline import stats
blas.scipy_openblas_set_num_threads64_.argtypes=[ctypes.c_int64];blas.scipy_openblas_set_num_threads64_(1)
torch.set_num_threads(3);torch.manual_seed(71);r={'cases':[],'replay':[]}
with torch.inference_mode():
 for t in (1,3,7,16,33,115,129,512):
  for h,kh in ((2,1),(8,2)):
   for maskkind in ('all','left','last'):
    d=32 if t!=115 else 256
    # Use noncontiguous BTC views, as in the live model.
    q=(torch.randn(1,t,h,d)*.7).bfloat16().transpose(1,2);k=torch.randn(1,t,kh,d).bfloat16().transpose(1,2);v=torch.randn_like(k)
    m=torch.ones(1,1,1,t,dtype=torch.bool)
    if maskkind=='left':m[...,:t//2]=False
    if maskkind=='last':m[...,:-1]=False
    expected=torch.nn.functional.scaled_dot_product_attention(q.float(),k.float().repeat_interleave(h//kh,1),v.float().repeat_interleave(h//kh,1),m,scale=d**-.5).transpose(1,2).bfloat16()
    saved=[x.clone() for x in (q,k,v,m)]
    for mode in (0,1):
     out=attention(q,k,v,m,d**-.5,mode);delta=(out.float()-expected.float()).abs().max().item()
     torch.testing.assert_close(out,expected,atol=.008,rtol=.008)
     if maskkind=='last':assert torch.equal(out,expected)
     r['cases'].append({'t':t,'heads':h,'mask':maskkind,'mode':mode,'max_delta':delta})
    assert all(torch.equal(x,y) for x,y in zip((q,k,v,m),saved))
 # Future visibility check.
 q=torch.ones(1,2,7,4).bfloat16();k=q[:,:1].clone();v=torch.zeros_like(k);m=torch.ones(1,1,1,7,dtype=torch.bool)
 a=attention(q,k,v,m,.5);v[:,:,-1]=7;b=attention(q,k,v,m,.5);assert not torch.equal(a[:,0],b[:,0])
 # Nonfinite inputs rejected, original wrapper can fall back.
 q[0,0,0,0]=float('nan')
 try:attention(q,k,v,m,.5);raise AssertionError('nonfinite accepted')
 except ValueError:pass
 r['future_visible']=True
 r['passed']=True
Path('/work/logs/cpu-attention-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('ATTENTION UNIT',len(r['cases']),flush=True)
