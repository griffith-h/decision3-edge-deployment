import torch,numpy as np
from pathlib import Path
from attention_math import Attention
torch.set_num_threads(3);root=Path('/work');items=torch.load(root/'attention-layers.pt',weights_only=False)
for i,item in enumerate(items):
    model=Attention(item,True).eval();x,cos,sin=(item[k].float() for k in ('x','cos','sin'))
    masks={'all':torch.zeros(1,1,1,115),'left':torch.cat([torch.full((1,1,1,32),-10000.),torch.zeros(1,1,1,83)],-1),'last_only':torch.cat([torch.full((1,1,1,114),-10000.),torch.zeros(1,1,1,1)],-1)}
    for label,mask in masks.items():
        with torch.inference_mode():ref=model(x,cos,sin,mask)
        np.savez(root/f'mask-{i}-{label}.npz',x=x.numpy(),cos=cos.numpy(),sin=sin.numpy(),mask=mask.numpy(),reference=ref.numpy())
print('EXPORTED 18 mask checks',flush=True)
