from loader import open_model
import json,torch
from pathlib import Path
model,engine,native,close=open_model()
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
original=qwen.causal_conv1d_fn;captured=[]
def capture(x,weight,bias=None,activation=None,**kwargs):
    captured.append({'x':x.detach().clone(),'w':weight.detach().clone(),'bias':None if bias is None else bias.detach().clone(),'activation':activation,'kwargs':kwargs,'stride':x.stride()})
    return original(x,weight,bias,activation=activation,**kwargs)
qwen.causal_conv1d_fn=capture
try:
    body=json.loads(Path('/base/request.json').read_text());model.system_one(state=body['state'],questions=body['questions'])
    torch.save(captured,'/work/conv-inputs.pt');print([(c['x'].shape,c['stride'],c['kwargs']) for c in captured],flush=True)
finally:qwen.causal_conv1d_fn=original;close()
