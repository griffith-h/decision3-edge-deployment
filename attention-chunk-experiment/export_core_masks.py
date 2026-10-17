from attention_math import Attention
from pathlib import Path
import torch,numpy as np
torch.set_num_threads(3);root=Path('/work');items=torch.load(root/'attention-layers.pt',weights_only=False)
with torch.inference_mode():
 for i,item in enumerate(items):
    m=Attention(item,True);x=item['x'].float();cos=item['cos'].float()[:,None];sin=item['sin'].float()[:,None]
    qg=m.linear(x,m.q_proj_weight).reshape(1,115,8,512);q,_=qg.chunk(2,-1);q=m.norm(q,m.q_norm_weight).transpose(1,2)
    k=m.norm(m.linear(x,m.k_proj_weight).reshape(1,115,2,256),m.k_norm_weight).transpose(1,2);v=m.linear(x,m.v_proj_weight).reshape(1,115,2,256).transpose(1,2)
    q=m.rope(q,cos,sin);k=m.rope(k,cos,sin)
    for label,nmasked in [('all',0),('left',32),('last_only',114)]:
        mask=torch.zeros(1,1,1,115);mask[...,:nmasked]=-10000
        ref=torch.nn.functional.scaled_dot_product_attention(q,k.repeat_interleave(4,1),v.repeat_interleave(4,1),attn_mask=mask,is_causal=False,scale=.0625).transpose(1,2).bfloat16().float()
        np.savez(root/f'core-mask-{i}-{label}.npz',q=q.contiguous().numpy(),k=k.contiguous().numpy(),v=v.contiguous().numpy(),mask=mask.numpy(),reference=ref.numpy())
print('EXPORTED 18 core mask checks',flush=True)
