import sys,torch,json
sys.path[:0]=['/release','/base','/work']
from scan import recurrent
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
torch.set_num_threads(3);torch.manual_seed(3576)
b,t,h,k,v=2,2,3,7,12
query=torch.randn(b,t,h,k*2,dtype=torch.bfloat16)[...,::2]*.03
key=torch.randn_like(query)*.03
value=torch.randn(b,t,h,v,dtype=torch.bfloat16)
g=-torch.rand(b,t,h)*.3;beta=torch.rand(b,t,h).to(torch.bfloat16)
args=(query,key,value,g,beta)
with torch.inference_mode():
    ref=qwen.torch_recurrent_gated_delta_rule(*args)[0]
    out=recurrent(*args)[0]
    qq,kk,vv,bb,gg=[x.transpose(1,2).to(torch.float32,memory_format=torch.contiguous_format) for x in (query,key,value,beta,g)]
    print('STRIDES',[(x.shape,x.stride()) for x in (qq,kk,vv,bb,gg,qq*k**-.5,gg.exp())])
    first=((qq[:,:,0]*k**-.5)*kk[:,:,0]).sum(-1)[...,None]*vv[:,:,0]*bb[:,:,0,None]
    print('FIRST',ref[:,0].flatten()[:20],out[:,0].flatten()[:20],first.to(torch.bfloat16).flatten()[:20])
    print('REF',ref.flatten()[:40]);print('OUT',out.flatten()[:40])
    print('DIFF',(out.float()-ref.float()).abs().max().item(),[torch.linalg.vector_norm(out[:,i].float()-ref[:,i].float()).item()/torch.linalg.vector_norm(ref[:,i].float()).item() for i in range(t)])
