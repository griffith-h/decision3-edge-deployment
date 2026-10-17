from loader import open_model,compare,stats,state,install
from variants import engine_with_library
import argparse,json,time,hashlib,statistics
from pathlib import Path
import torch
from conv_btc import install as install_conv
from prepared_scan import install as install_prepare
from setup_helpers import cases
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['conv','conv-prepare','conversion-conv-prepare'],required=True);p.add_argument('--stage',choices=['paired','single'],required=True);p.add_argument('--count',type=int,default=20);a=p.parse_args()
model,baseline,native,close=open_model();baseline.restore_forwards();baseline.restore_forwards=None
candidate=engine_with_library('/work/libconversion_bridge.so') if a.variant.startswith('conversion') else baseline
modules=list(model.backbone.modules());cache={};current=None;restores=[];counts={};base_conv=qwen.causal_conv1d_fn;base_scan=qwen.torch_chunk_gated_delta_rule

def retire():
    global current,restores
    if current is not None:
        cache[id(current)]={id(m):m._bp_handle for m in modules if hasattr(m,'_bp_handle')}
        current.restore_forwards();current.restore_forwards=None;current=None
    for restore in reversed(restores):restore()
    restores=[]
    assert qwen.causal_conv1d_fn is base_conv and qwen.torch_chunk_gated_delta_rule is base_scan

def use(name):
    global current,counts
    retire();counts={}
    if name=='candidate':
        restore,c=install_conv();restores.append(restore);counts['conv']=c
        if 'prepare' in a.variant:
            restore,c=install_prepare();restores.append(restore);counts['prepare']=c
    engine=baseline if name=='baseline' else candidate
    engine.groups=[];engine.restore_forwards=install(model,engine)
    for m in modules:
        if hasattr(m,'_bp_handle'):m._bp_handle=cache.get(id(engine),{}).get(id(m))
    current=engine;torch.set_num_threads(3)
allcases=cases();single=next(c for c in allcases if c['id']=='single');refs=json.loads(Path('/projection/logs/scatter-balanced-none.json').read_text())['cases'];fp32=json.loads(Path('/remainder/logs/model-results.json').read_text())['baseline_cases']
def invoke(c=single):return model.system_one(**{k:c[k] for k in ('state','questions','images','videos') if k in c})
def check(r,c=single):
    x=compare(r,refs[c['id']]['response']);y=compare(r,fp32[c['id']]);assert x['passed'] and y['passed'],(c['id'],x,y)
    return {'comparison':x,'comparison_vs_fp32':y}
result={'variant':a.variant,'stage':a.stage,'settings':state(),'records':[],'cases':{},'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').iterdir() if p.suffix in ('.py','.c','.cpp','.so')}}
path=Path('/work/logs/final-'+a.stage+'-'+a.variant+'.json')
def save():path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
try:
    if a.stage=='paired':
        for name in ('baseline','candidate'):
            use(name);invoke();check(invoke())
        for i in range(a.count):
            for name in (['baseline','candidate'] if i%2==0 else ['candidate','baseline']):
                use(name);invoke();t=time.perf_counter();r=invoke();elapsed=time.perf_counter()-t
                checked=check(r)
                assert name=='baseline' or all(c['optimized']==12 and c['fallback']==0 for c in counts.values()),counts
                result['records'].append({'round':i,'variant':name,'seconds':elapsed,'cpu_counts':counts,**checked})
                print('PAIR',i,name,elapsed,checked,flush=True)
        result['timing']={name:stats([r['seconds'] for r in result['records'] if r['variant']==name]) for name in ('baseline','candidate')}
        pairs=[{r['variant']:r['seconds'] for r in result['records'] if r['round']==i} for i in range(a.count)]
        result['paired_reduction_median']=statistics.median(1-r['candidate']/r['baseline'] for r in pairs);result['faster_rounds']=sum(r['candidate']<r['baseline'] for r in pairs)
        use('baseline');after=invoke();result['baseline_after']=check(after)
        result['scope']='Same model; persistent engines; alternating order; one untimed warmup before each measured trial; two engines only when bridge differs. No HTTP. Not same resource condition as single-engine100.'
    else:
        use('candidate');invoke();invoke();times=[]
        for i in range(a.count):
            t=time.perf_counter();r=invoke();elapsed=time.perf_counter()-t;checked=check(r);times.append(elapsed)
            result['records'].append({'index':i,'seconds':elapsed,**checked});print('SINGLE',i,elapsed,checked,flush=True)
        result['timing']=stats(times)
        for c in allcases:
            r=invoke(c);result['cases'][c['id']]={'response':r,**check(r,c)};save()
        result['cpu_counts']=counts;result['scope']='Single candidate engine; warmed115-token one-question system_one; no HTTP; every response checked. Seven cases/fourteen questions are numerical regression, not labeled task accuracy.'
    result.update(passed=True,handles={'baseline':len(baseline.handles),'candidate':len(candidate.handles)},engine_stats=dict(candidate.stats),final_settings=state());save()
finally:
    retire()
    if candidate is not baseline:candidate.close()
    close()
