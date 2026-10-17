from baseline import load_baseline
from pathlib import Path
import json,time,torch,inspect
model,engine,counts,close=load_baseline()
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
Path('/work/logs/attention-reference-source.txt').write_text(inspect.getsource(qwen.Qwen3_5Attention)+inspect.getsource(qwen.apply_rotary_pos_emb))
body=json.loads(Path('/base/request.json').read_text());model.system_one(state=body['state'],questions=body['questions'])
items=[];restores=[]
for name,m in model.backbone.named_modules():
    if type(m) is not qwen.Qwen3_5Attention:continue
    original=m.forward
    def forward(*args,_original=original,_m=m,_name=name,**kw):
        before=time.perf_counter();out=_original(*args,**kw);elapsed=time.perf_counter()-before
        items.append({'name':_name,'x':(args[0] if args else kw['hidden_states']).detach().clone(),'cos':kw['position_embeddings'][0].detach().clone(),'sin':kw['position_embeddings'][1].detach().clone(),'mask':kw['attention_mask'].detach().clone(),
                      'weights':{k:v.detach().clone() for k,v in _m.state_dict().items()},'out':out[0].detach().clone(),'seconds':elapsed,'eps':_m.q_norm.eps,'scaling':_m.scaling,'head_dim':_m.head_dim})
        return out
    m.forward=forward;restores.append((m,original))
try:
    body=json.loads(Path('/base/request.json').read_text());r=model.system_one(state=body['state'],questions=body['questions']);assert len(items)==6
    torch.save(items,'/work/attention-layers.pt');print([(i['name'],i['x'].shape,i['cos'].shape,i['mask'].shape,i['seconds']) for i in items],flush=True)
finally:
    for m,o in restores:m.forward=o
    close()
