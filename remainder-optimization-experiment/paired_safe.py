"""Final guarded candidate vs FP32 output, same model, two warm engines."""
import argparse,json,time,statistics
from pathlib import Path
import torch
from common import load,compare,stats,sources,state
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
model,safe,native,close=load(a.model,half_output=True,safe_output=True)
from adapter import Engine,install
fp32=Engine(fusion=True,parallel=True);fp32.restore_forwards=None
current=safe;cache={};modules=list(model.backbone.modules())
def retire():
    global current
    if current is None:return
    cache[id(current)]={id(m):m._bp_handle for m in modules if hasattr(m,'_bp_handle')}
    current.restore_forwards();current.restore_forwards=None;current=None
def use(target):
    global current
    retire();target.groups=[];target.restore_forwards=install(model,target)
    for m in modules:
        if hasattr(m,'_bp_handle'):m._bp_handle=cache.get(id(target),{}).get(id(m))
    current=target
body=json.loads(Path('/base/request.json').read_text())
def invoke():return model.system_one(state=body['state'],questions=body['questions'])
result={'scope':'Same loaded model, two persistent78-handle engines, final guarded FP16 output vs original FP32 output. Six alternating rounds, per-trial Torch3 and explicit warmup. No HTTP/server. Separate resource condition from single-engine100.',
        'sources':sources(),'settings':state()}
try:
    use(fp32);invoke();reference=invoke()
    use(safe);invoke();assert compare(invoke(),reference)['passed']
    records=[]
    for i in range(6):
        for name,target in ([('fp32',fp32),('safe_fp16',safe)] if i%2==0 else [('safe_fp16',safe),('fp32',fp32)]):
            use(target);torch.set_num_threads(3);invoke()
            t=time.perf_counter();response=invoke();elapsed=time.perf_counter()-t
            comp=compare(response,reference);assert comp['passed']
            records.append({'round':i,'variant':name,'seconds':elapsed,'comparison':comp})
            print('SAFE_PAIR',i,name,elapsed,flush=True)
    result.update(records=records,summary={n:stats([r['seconds'] for r in records if r['variant']==n]) for n in ('fp32','safe_fp16')},
                  safe_stats=safe.stats,handle_counts={'fp32':len(fp32.handles),'safe_fp16':len(safe.handles)},passed=True)
    assert result['handle_counts']=={'fp32':78,'safe_fp16':78}
    Path('/work/logs/safe-paired.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
finally:retire();fp32.close();close()
