from baseline import load_baseline,compare,stats,state,cases
from chunk_blas import install
import json,time,hashlib,statistics,torch
from pathlib import Path
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
model,engine,baseline_counts,close=load_baseline();base_fn=qwen.torch_chunk_gated_delta_rule;restore=None;counts={}
allcases=cases();single=next(c for c in allcases if c['id']=='single')
refs=json.loads(Path('/previous/logs/final-single-conversion-conv-prepare.json').read_text())['cases'];fp32=json.loads(Path('/remainder/logs/model-results.json').read_text())['baseline_cases']
def invoke(c=single):return model.system_one(**{k:c[k] for k in ('state','questions','images','videos') if k in c})
def check(r,c=single):
    a=compare(r,refs[c['id']]['response']);b=compare(r,fp32[c['id']]);assert a['passed'] and b['passed'],(c['id'],a,b)
    return {'vs_454':a,'vs_fp32':b}
def use(name):
    global restore,counts
    if restore:restore();restore=None
    assert qwen.torch_chunk_gated_delta_rule is base_fn
    if name=='chunk':restore,counts=install(32,64)
    else:counts={}
    torch.set_num_threads(3)
result={'settings':state(),'records':[],'cases':{},'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').iterdir() if p.suffix in ('.py','.c','.cpp','.so')}}
def save():Path('/work/logs/chunk-model.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
try:
    invoke();check(invoke());use('chunk');invoke();check(invoke())
    for i in range(20):
        for name in (['baseline','chunk'] if i%2==0 else ['chunk','baseline']):
            use(name);invoke();t=time.perf_counter();r=invoke();elapsed=time.perf_counter()-t;comp=check(r)
            assert name=='baseline' or counts=={'optimized':12,'fallback':0},counts
            result['records'].append({'round':i,'variant':name,'seconds':elapsed,**comp});print('PAIR',i,name,elapsed,comp,flush=True)
    result['timing']={n:stats([r['seconds'] for r in result['records'] if r['variant']==n]) for n in ('baseline','chunk')}
    use('chunk')
    for c in allcases:
        r=invoke(c);result['cases'][c['id']]={'response':r,**check(r,c)};save()
    use('baseline');result['baseline_after']=check(invoke());result.update(passed=True,engine_stats=engine.stats,handles=len(engine.handles));save()
finally:
    if restore:restore()
    close()
