"""Paired baseline/fusion/parallel/combined model checks, no listening endpoint."""
import argparse,copy,hashlib,json,math,os
from pathlib import Path
import statistics,sys,time
import torch
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--count',type=int,default=30);a=p.parse_args()
sys.path[:0]=['/release','/base','/work',a.model]
from test_units import run as units
units()
from d3_runtime import D3
from cpu_optimized_server import optimize
from rknn_linear import Engine as OriginalEngine,install as install_original,uninstall
from cpu_attention_optimized import install_conv,install_scan,install_sdpa
from native_norm import install_norm,install_native_conv
from adapter import Engine,install
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

def numeric(value,path=''):
    out={}
    if isinstance(value,dict):
        for k,v in value.items():
            if k not in ('confidence','legend'):out.update(numeric(v,path+'/'+k))
    elif isinstance(value,(int,float)):out[path]=value
    return out

def compare(actual,expected):
    x,y=numeric(actual['answers']),numeric(expected['answers']);assert x.keys()==y.keys()
    delta=max(abs(x[k]-y[k]) for k in x)
    choices=all(v.get('choice')==expected['answers'][k].get('choice') for k,v in actual['answers'].items())
    return {'max_probability_or_score_delta':delta,'choices_unchanged':choices,'passed':delta<=.01 and choices}

cases=json.loads(Path('/references/request-cases.json').read_text())
image=json.loads(Path('/references/golden-reference.json').read_text())['image'];image['state']='Describe the attached image.'
cases.append({'id':'image',**image})
single=next(c for c in cases if c['id']=='single')
for name,text in [('novel-math','请计算 37 乘以 19，并检查结果。'),('novel-chat','今天下班后想和朋友散步聊天，有什么轻松的话题？')]:
    item=copy.deepcopy(single);item['id']=name;item['state']=text;cases.append(item)
cpu=json.loads(Path('/references/logs/benchmark-all.json').read_text())['cases']
torch.set_num_threads(3)
model=D3.from_pretrained(a.model,device='cpu',verify='full');model.model_name='vllm-sr/d3-edge'
assert model.attention_mode=='noncausal_full_attention' and len(model.encoder.stack)==12
optimize(model);native={}
restore_native=[install_conv(),install_scan(128),install_sdpa(),install_norm(model.backbone,native),install_native_conv(native)]
result={'scope':'Isolated three-thread process, fixed official model/12 layers/masks and CPU kernels. 115-token single-question timings, concurrency1, initialization excluded, no response cache.',
 'settings':{'torch_threads':torch.get_num_threads(),'affinity':sorted(os.sched_getaffinity(0)),'OMP_THREAD_LIMIT':os.environ.get('OMP_THREAD_LIMIT')},
 'source_sha256':{str(path.relative_to('/work')):hashlib.sha256(path.read_bytes()).hexdigest() for path in Path('/work').glob('*') if path.suffix in ('.py','.cpp')},
 'baseline_cases':{},'variants':{}}
output=Path('/work/logs/model-results.json')
def save():output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
def invoke(case):return model.system_one(**{k:case[k] for k in ('state','questions','images','videos') if k in case})
def timing(count):
    invoke(single);values=[];response=None
    for i in range(count):
        t=time.perf_counter();response=invoke(single);values.append(time.perf_counter()-t)
        if (i+1)%5==0:print('TIMING',variant,i+1,round(values[-1],6),flush=True)
    return {'count':count,'seconds':values,'median':statistics.median(values),'p95_nearest_rank':sorted(values)[math.ceil(.95*count)-1],
            'max':max(values),'min':min(values),'below_one_second':sum(t<1 for t in values),'response':response}
try:
    for variant,fusion,parallel in [('baseline',False,False),('fusion',True,False),('parallel',False,True),('combined',True,True),('baseline_recheck',False,False)]:
        engine=None;restore_adapter=None
        try:
            if variant.startswith('baseline'):
                engine=OriginalEngine();install_original(model,engine);restore_adapter=lambda:uninstall(model)
            else:
                engine=Engine(fusion=fusion,parallel=parallel);restore_adapter=install(model,engine)
            record={'fusion':fusion,'parallel':parallel,'cases':{}}
            result['variants'][variant]=record
            record['timing']=timing(a.count if variant!='baseline_recheck' else 15);save()
            selected_cases=[single] if variant=='baseline_recheck' else cases
            for case in selected_cases:
                t=time.perf_counter();response=invoke(case);elapsed=time.perf_counter()-t
                if variant=='baseline':result['baseline_cases'][case['id']]=response
                comparison=compare(response,result['baseline_cases'][case['id']])
                entry={'seconds':elapsed,'vs_three_thread_baseline':comparison,'response':response}
                if case['id'] in cpu:entry['vs_historical_cpu']=compare(response,cpu[case['id']]['cpu']['response'])
                record['cases'][case['id']]=entry;save()
                print('CASE',variant,case['id'],elapsed,json.dumps(comparison),flush=True)
            record['passed']=all(x['vs_three_thread_baseline']['passed'] for x in record['cases'].values())
            record['engine_calls']=engine.calls;record['cpu_fallbacks']=engine.cpu_fallbacks
            if isinstance(engine,Engine):record['group_stats']=engine.stats;record['groups']=engine.groups
            save()
        finally:
            if restore_adapter is not None:restore_adapter()
            if engine is not None:engine.close()
    candidates=[(name,entry) for name,entry in result['variants'].items() if not name.startswith('baseline') and entry['passed']]
    if candidates:
        best=min(candidates,key=lambda item:item[1]['timing']['median'])[0]
        fusion=result['variants'][best]['fusion'];parallel=result['variants'][best]['parallel']
        variant='selected_'+best
        engine=Engine(fusion=fusion,parallel=parallel);restore_adapter=install(model,engine)
        try:
            selected_timing=timing(100)
            selected_timing['comparison']=compare(selected_timing['response'],result['baseline_cases']['single'])
            selected_timing['variant']=best
            result['selected_100']=selected_timing;result['selected_stats']=engine.stats
            assert selected_timing['comparison']['passed']
            save()
        finally:restore_adapter();engine.close()
    result['native_stats']=native;result['passed']=all(v['passed'] for v in result['variants'].values());save()
    print('EXPERIMENT_COMPLETE',result['passed'],flush=True)
finally:
    for callback in reversed(restore_native):callback()
