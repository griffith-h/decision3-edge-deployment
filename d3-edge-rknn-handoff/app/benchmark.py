"""Paired model test with unchanged D3 PromptEncoder and optimized CPU baseline."""
import argparse
import hashlib
import inspect
import json
import statistics
import sys
import time
from pathlib import Path
import torch
from rknn_linear import Engine, install, uninstall
from cpu_optimized_server import optimize

parser=argparse.ArgumentParser()
parser.add_argument('--model',required=True)
parser.add_argument('--repeats',type=int,default=3)
parser.add_argument('--selection',choices=['all','mlp'],default='all')
args=parser.parse_args()
root=Path('/work')
sys.path.insert(0,args.model)
from d3_runtime import D3, PromptEncoder
torch.set_num_threads(4)
model=D3.from_pretrained(args.model,device='cpu',verify='full')
model.model_name='vllm-sr/d3-edge'
assert model.attention_mode=='noncausal_full_attention'
assert isinstance(model.encoder,PromptEncoder)
optimized,size=optimize(model)
cases=json.loads((root/'request-cases.json').read_text())
single=next(x for x in cases if x['id']=='single')
cases=[single]+[x for x in cases if x['id']!='single']
image=json.loads((root/'golden-reference.json').read_text())['image']
image['state']='Describe the attached image.'
cases.append({'id':'image',**image})
report={'attention_mode':model.attention_mode,'prompt_encoder_sha256':hashlib.sha256(inspect.getsource(PromptEncoder).encode()).hexdigest(),
        'torch':torch.__version__,'threads':torch.get_num_threads(),'batch_size':model.batch_size,
        'cpu_optimized_modules':optimized,'selection':args.selection,'repeats':args.repeats,'cases':{},'passed':False}
destination=root/'logs'/f'benchmark-{args.selection}.json'
def save():
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
def invoke(case):
    return model.system_one(**{k:case[k] for k in ['state','questions','images','videos'] if k in case})
def numbers(value,path=''):
    result={}
    if isinstance(value,dict):
        for key,child in value.items():
            if key not in ('confidence','legend'):
                result.update(numbers(child,path+'/'+key))
    elif isinstance(value,(int,float)):
        result[path]=value
    return result

profile={}
original=[]
for name,module in model.backbone.named_modules():
    if isinstance(module,(torch.nn.Linear,torch.nn.Conv1d,torch.nn.Conv2d,torch.nn.Conv3d)):
        forward=module.forward
        def timed(inputs,*positional,_forward=forward,_name=name,**keywords):
            start=time.perf_counter(); out=_forward(inputs,*positional,**keywords)
            p=profile.setdefault(_name,{'seconds':0,'calls':0,'shape':list(inputs.shape)})
            p['seconds']+=time.perf_counter()-start; p['calls']+=1
            return out
        original.append((module,forward)); module.forward=timed
start=time.perf_counter(); invoke(single); elapsed=time.perf_counter()-start
for module,forward in original:
    module.forward=forward
report['cpu_profile']={'total_seconds':elapsed,'linear_conv_seconds':sum(p['seconds'] for p in profile.values()),'modules':profile}
save()
print('CPU_PROFILE',json.dumps({k:v for k,v in report['cpu_profile'].items() if k!='modules'}),flush=True)
engine=None
try:
    for mode in ['cpu','npu']:
        if mode=='npu':
            engine=Engine()
            report['offloaded_modules']=install(model,engine,args.selection)
            print('OFFLOAD',len(report['offloaded_modules']),flush=True)
        for case in cases:
            # Warm up this shape; first NPU request includes persistent weight/context setup.
            start=time.perf_counter(); warm=invoke(case); warm_seconds=time.perf_counter()-start
            timings=[]
            before_calls=engine.calls if engine else 0
            before_time=engine.seconds if engine else 0
            before_fallback=engine.cpu_fallbacks if engine else 0
            for _ in range(args.repeats):
                start=time.perf_counter(); response=invoke(case); timings.append(time.perf_counter()-start)
                assert response==warm, 'Repeated response changed within the same backend'
            record={'warmup_seconds':warm_seconds,'seconds':timings,'median_seconds':statistics.median(timings),'response':response}
            if mode=='npu':
                record.update(npu_calls=engine.calls-before_calls,npu_adapter_seconds=engine.seconds-before_time,
                              cpu_fallback_calls=engine.cpu_fallbacks-before_fallback)
                baseline=report['cases'][case['id']]['cpu']
                a=numbers(response['answers']); b=numbers(baseline['response']['answers'])
                assert set(a)==set(b)
                delta=max(abs(a[key]-b[key]) for key in a)
                matches=all(v.get('choice')==baseline['response']['answers'][key].get('choice') for key,v in response['answers'].items())
                record.update(max_probability_or_score_delta=delta,choices_unchanged=matches,
                              speedup=baseline['median_seconds']/record['median_seconds'],parity_passed=delta<=0.01 and matches)
                assert model.attention_mode=='noncausal_full_attention' and isinstance(model.encoder,PromptEncoder)
            report['cases'].setdefault(case['id'],{})[mode]=record
            save()
            print('CASE',mode,case['id'],json.dumps({k:v for k,v in record.items() if k!='response'}),flush=True)
    report['passed']=all(c['npu']['parity_passed'] for c in report['cases'].values())
    report['setup_seconds']=engine.create_seconds
    report['npu_handles']=len(engine.handles)
    save()
    assert report['passed'], 'CPU/NPU probability parity failed; do not promote'
    print('PASS: same choices; numeric delta <= 0.01; original attention code unchanged.',flush=True)
finally:
    if engine:
        uninstall(model)
        engine.close()
