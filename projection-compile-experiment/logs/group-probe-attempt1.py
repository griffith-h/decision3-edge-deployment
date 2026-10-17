"""Replay real captured inputs; distinguish per-core execution from output-copy cost."""
import setup_helpers,json,time,statistics
from pathlib import Path
import torch
from common import load,stats
from scattered_adapter import PackedEngine
from adapter import Engine
snapshot='/models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e'
model,base,native,close=load(snapshot,safe_output=True);captured={};hooks=[];events=[];other=[]
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
for parent in model.backbone.modules():
    kind='full' if type(parent) is qwen.Qwen3_5Attention else 'delta' if type(parent) is qwen.Qwen3_5GatedDeltaNet else None
    if kind is None or kind in captured:continue
    names=['q_proj','k_proj','v_proj'] if kind=='full' else ['in_proj_qkv','in_proj_z']
    modules=[getattr(parent,n) for n in names];captured[kind]={'modules':modules}
    def hook(m,args,kind=kind):
        if 'input' not in captured[kind]:captured[kind]['input']=args[0].detach().clone()
    hooks.append(modules[0].register_forward_pre_hook(hook))
old=Engine.execute
def traced(self,m,x,*args,**kw):
    start=time.perf_counter();out=old(self,m,x,*args,**kw)
    events.append({'core':self.core.get(id(m),0),'n':m.out_features,'start':start,'end':time.perf_counter()});return out
result={'scope':'Isolated two-fork replay from real warmed model inputs, one group at a time. Per-piece wall includes SDK sync and output conversion, not pure NPU compute. Instrumented numbers are diagnostic, not end-to-end speed.','groups':{}}
try:
    body=json.loads(Path('/base/request.json').read_text());model.system_one(state=body['state'],questions=body['questions'])
    for h in hooks:h.remove()
    Engine.execute=traced
    candidates={n:PackedEngine(n) for n in ('merge','balanced')};other=list(candidates.values())
    for e in other:e.profile=True
    with torch.inference_mode():
        for kind,data in captured.items():
            modules=data['modules'];x=data['input'];reference=base.group(modules,x)
            variants={'baseline':base,**candidates};records=[]
            for e in variants.values():e.group(modules,x)
            for i in range(20):
                names=list(variants)
                if i%2:names.reverse()
                for name in names:
                    e=variants[name];events.clear()
                    if hasattr(e,'events'):e.events.clear()
                    start=time.perf_counter();outputs=e.group(modules,x);elapsed=time.perf_counter()-start
                    trace=events[:] if name=='baseline' else e.events[:]
                    deltas=[(outputs[id(m)].float()-reference[id(m)].float()).abs().max().item() for m in modules]
                    assert all(torch.isfinite(outputs[id(m)]).all() for m in modules)
                    records.append({'round':i,'variant':name,'seconds':elapsed,'pieces':[{'core':v['core'],'n':v['n'],'seconds':v['end']-v['start']} for v in trace],'max_tensor_delta':max(deltas)})
            result['groups'][kind]={'input_shape':list(x.shape),'original_n':[m.out_features for m in modules],
                'records':records,'summary':{n:stats([r['seconds'] for r in records if r['variant']==n]) for n in variants}}
    Path('/work/logs/group-probe.json').write_text(json.dumps(result,indent=2)+'\n');print('GROUP_PROBE_DONE',flush=True)
finally:
    Engine.execute=old
    for e in other:e.close()
    close()
