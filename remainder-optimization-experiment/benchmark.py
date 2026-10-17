"""CPU math ablations, then public SDK output-type precision experiment."""
import argparse,copy,ctypes,json,statistics,time
from pathlib import Path
from common import load,compare,stats,sources,state
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
control={};model,engine,native,close=load(a.model,control=control)
from test_math import run as units
from adapter import Engine,install
import torch
cases=json.loads(Path('/references/request-cases.json').read_text())
image=json.loads(Path('/references/golden-reference.json').read_text())['image'];image['state']='Describe the attached image.';cases.append({'id':'image',**image})
single=next(c for c in cases if c['id']=='single')
for name,text in [('novel-math','请计算 37 乘以 19，并检查结果。'),('novel-chat','今天下班后想和朋友散步聊天，有什么轻松的话题？')]:
    item=copy.deepcopy(single);item['id']=name;item['state']=text;cases.append(item)
def invoke(case):return model.system_one(**{k:case[k] for k in ('state','questions','images','videos') if k in case})
variants={'baseline':{},'norm':{'norm':True},'activation':{'activation':True},'all':{'norm':True,'activation':True,'mlp':True}}
result={'scope':'Independent process, frozen native scan+combined bridge baseline; CPU math ablation uses same model/engine. FP16 output is a separate numerical experiment with original FP16 weights, not quantization.',
        'settings':state(),'sources':sources(),'baseline_cases':{},'cpu':{},'case_count':len(cases),'question_count':sum(len(c['questions']) for c in cases)}
output=Path('/work/logs/model-results.json')
def save():output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
def activate_math(name):control.clear();control.update(variants[name])
def timing_check(name):
    torch.set_num_threads(3);invoke(single)
    t=time.perf_counter();response=invoke(single);elapsed=time.perf_counter()-t
    comp=compare(response,result['baseline_cases']['single']);assert comp['passed'],(name,comp)
    return {'seconds':elapsed,'comparison':comp}
half_engine=None;current=engine;cache={}
modules=list(model.backbone.modules())
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
try:
    result['units']=units()
    activate_math('all')
    import torch.nn.functional as F
    x=torch.randn(17)
    torch.testing.assert_close(F.silu(x),x*torch.sigmoid(x))
    y=torch.randn(17,dtype=torch.bfloat16);original=y.clone()
    assert F.silu(y,inplace=True).data_ptr()==y.data_ptr()
    torch.testing.assert_close(y,(original.float()*torch.sigmoid(original.float())).bfloat16())
    z=torch.randn(17,dtype=torch.bfloat16,requires_grad=True)
    F.silu(z).sum().backward();assert z.grad is not None
    out=torch.empty(17);assert torch.sigmoid(x,out=out).data_ptr()==out.data_ptr()
    result['wrapper_fallback_checks_passed']=True
    activate_math('baseline');invoke(single)
    for case in cases:
        torch.set_num_threads(3)
        result['baseline_cases'][case['id']]=invoke(case)
    save()
    for name in ('norm','activation','all'):
        activate_math(name);record={'cases':{}}
        for case in (cases if name=='all' else [single]):
            t=time.perf_counter();response=invoke(case);elapsed=time.perf_counter()-t
            comp=compare(response,result['baseline_cases'][case['id']])
            record['cases'][case['id']]={'seconds':elapsed,'comparison':comp,'response':response}
            print('CASE',name,case['id'],elapsed,comp,flush=True)
        record['passed']=all(r['comparison']['passed'] for r in record['cases'].values())
        result['cpu'][name]=record;save()
    acceptable=[n for n in ('norm','activation','all') if result['cpu'][n]['passed']]
    paired=[]
    for i in range(12):
        order=['baseline']+acceptable
        if i%2:order.reverse()
        for name in order:
            activate_math(name);r=timing_check(name);r.update(round=i,variant=name);paired.append(r)
            print('PAIR',i,name,r['seconds'],flush=True)
    result['cpu_paired']={'records':paired,'summary':{n:stats([r['seconds'] for r in paired if r['variant']==n]) for n in ['baseline']+acceptable}};save()
    # Only a candidate with all seven cases qualifies for the full stability test.
    best='all' if result['cpu']['all']['passed'] and result['cpu_paired']['summary']['all']['median']<result['cpu_paired']['summary']['baseline']['median'] else 'baseline'
    activate_math(best);values=[];deltas=[]
    for i in range(100):
        t=time.perf_counter();response=invoke(single);values.append(time.perf_counter()-t)
        comp=compare(response,result['baseline_cases']['single']);assert comp['passed'];deltas.append(comp['max_delta'])
        if (i+1)%10==0:print('STABLE_CPU',i+1,values[-1],flush=True)
    result['cpu_100']={'variant':best,**stats(values),'max_delta':max(deltas),'every_response_checked':True};save()
    # Allocate the second engine only after the single-engine CPU experiment.
    half_engine=Engine(fusion=True,parallel=True)
    old=half_engine.lib;lib=ctypes.CDLL('/work/libhalf_bridge.so')
    for name in ('mm_error','mm_create_core','mm_run','mm_destroy','mm_prepare_bf16','mm_copy_prepared','mm_run_prepared_bf16','mm_run_bf16'):
        src=getattr(old,name);dst=getattr(lib,name);dst.argtypes=src.argtypes;dst.restype=src.restype
    half_engine.lib=lib;half_engine.restore_forwards=None
    result['half_output']={'cases':{}}
    try:
        use(half_engine);invoke(single)
        for case in cases:
            t=time.perf_counter();response=invoke(case);elapsed=time.perf_counter()-t
            comp=compare(response,result['baseline_cases'][case['id']])
            result['half_output']['cases'][case['id']]={'seconds':elapsed,'comparison':comp,'response':response}
            print('HALF_CASE',case['id'],elapsed,comp,flush=True);save()
        passed=all(r['comparison']['passed'] for r in result['half_output']['cases'].values())
        result['half_output']['passed']=passed;save()
        if passed:
            paired=[]
            for i in range(12):
                for name,target in ([('fp32',engine),('fp16',half_engine)] if i%2==0 else [('fp16',half_engine),('fp32',engine)]):
                    use(target);r=timing_check(name);r.update(round=i,variant=name);paired.append(r)
                    print('HALF_PAIR',i,name,r['seconds'],flush=True)
            result['half_output']['paired']={'records':paired,'summary':{n:stats([r['seconds'] for r in paired if r['variant']==n]) for n in ('fp32','fp16')}};save()
    except Exception as exc:
        result['half_output'].update(passed=False,error=repr(exc));save()
    result['passed']=result['cpu']['all']['passed'];save()
finally:
    retire()
    if half_engine is not None:half_engine.close()
    close()
