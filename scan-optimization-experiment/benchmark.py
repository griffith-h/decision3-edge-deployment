"""Scan microbenchmark, same-engine paired end-to-end and accuracy tests."""
import argparse,copy,hashlib,json,math,os,statistics,sys,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--stage',default='micro');a=p.parse_args()
sys.path[:0]=['/release','/base','/combined','/work',a.model]
from d3_runtime import D3
from cpu_optimized_server import optimize
from cpu_attention_optimized import install_conv,install_scan,install_sdpa
from native_norm import install_norm,install_native_conv
from adapter import Engine,install
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from scan import make_scan

def numeric(value,path=''):
    out={}
    if isinstance(value,dict):
        for k,v in value.items():
            if k not in ('confidence','legend'):out.update(numeric(v,path+'/'+k))
    elif isinstance(value,(int,float)):out[path]=value
    return out
def compare(actual,expected):
    x,y=numeric(actual['answers']),numeric(expected['answers']);assert x.keys()==y.keys()
    assert all(math.isfinite(v) for v in x.values())
    delta=max(abs(x[k]-y[k]) for k in x)
    choices=all(v.get('choice')==expected['answers'][k].get('choice') for k,v in actual['answers'].items())
    return {'max_probability_or_score_delta':delta,'choices_unchanged':choices,'passed':delta<=.01 and choices}
def stats(values):
    return {'count':len(values),'median':statistics.median(values),'p95_nearest_rank':sorted(values)[math.ceil(.95*len(values))-1],
            'min':min(values),'max':max(values),'seconds':values}
def operating_state():
    paths=['/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq','/sys/class/devfreq/27700000.npu/cur_freq']
    paths.extend(str(p) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp'))
    return {p:Path(p).read_text().strip() for p in paths if Path(p).is_file()}
def tensor_diff(x,y):
    d=(x.float()-y.float()).abs()
    return {'bitwise_equal':torch.equal(x,y),'max_abs':d.max().item(),'rms_abs':d.square().mean().sqrt().item(),
            'relative_l2':(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(y.float()).clamp_min(1e-30)).item(),
            'finite':bool(torch.isfinite(x).all())}
torch.set_num_threads(3)
model=D3.from_pretrained(a.model,device='cpu',verify='full');model.model_name='vllm-sr/d3-edge'
assert model.attention_mode=='noncausal_full_attention' and len(model.encoder.stack)==12
optimize(model);native={}
restores=[install_conv(),install_scan(128),install_sdpa(),install_norm(model.backbone,native),install_native_conv(native)]
engine=Engine(fusion=True,parallel=True);restore_engine=install(model,engine)
baseline=qwen.torch_chunk_gated_delta_rule
cases=json.loads(Path('/references/request-cases.json').read_text())
image=json.loads(Path('/references/golden-reference.json').read_text())['image'];image['state']='Describe the attached image.'
cases.append({'id':'image',**image})
single=next(c for c in cases if c['id']=='single')
for name,text in [('novel-math','请计算 37 乘以 19，并检查结果。'),('novel-chat','今天下班后想和朋友散步聊天，有什么轻松的话题？')]:
    item=copy.deepcopy(single);item['id']=name;item['state']=text;cases.append(item)
def invoke(case):return model.system_one(**{k:case[k] for k in ('state','questions','images','videos') if k in case})
variants={'baseline':baseline,'compact':make_scan(baseline,'compact')}
for tile in (16,32,64,128):
    for fused in (0,1,2,3,4,5):variants[f'native_t{tile}_f{fused}']=make_scan(baseline,tile=tile,fused=fused)
result={'stage':a.stage,'scope':'One combined fusion+parallel engine, same original model and all other CPU kernels. Independent process, no listening endpoint. Timings exclude initialization/HTTP; concurrency 1.',
        'settings':{'torch_threads':torch.get_num_threads(),'affinity':sorted(os.sched_getaffinity(0)),'omp_limit':os.environ.get('OMP_THREAD_LIMIT'),
                    'initial_operating_state':operating_state(),'torch_version':torch.__version__},
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('/work','/combined') for p in Path(folder).iterdir() if p.suffix in ('.py','.c','.cpp','.so')},
        'micro':{},'baseline_cases':{},'model':{}}
out=Path('/work/logs/'+a.stage+'-results.json')
def save():out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
try:
    captured=[]
    def capture(*args,**kwargs):
        output=baseline(*args,**kwargs);captured.append((args,kwargs,output[0]));return output
    qwen.torch_chunk_gated_delta_rule=capture
    result['capture_response']=invoke(single)
    assert len(captured)==6
    qwen.torch_chunk_gated_delta_rule=baseline
    result['captured_metadata']=[{'args':[{'shape':list(x.shape),'dtype':str(x.dtype)} for x in args],
        'kwargs':{k:({'shape':list(v.shape),'dtype':str(v.dtype)} if isinstance(v,torch.Tensor) else repr(v)) for k,v in kw.items()}} for args,kw,_ in captured]
    torch.save(captured,'/work/logs/captured-scans.pt');save()
    for name,fn in (variants.items() if a.stage=='micro' else []):
        record=[]
        for i,(args,kwargs,ref) in enumerate(captured):
            values=[]
            with torch.inference_mode():
                for _ in range(7):
                    t=time.perf_counter();actual=fn(*args,**kwargs)[0];values.append(time.perf_counter()-t)
            d=tensor_diff(actual,ref);assert d['finite']
            record.append({'scan':i,'shapes':[list(x.shape) for x in args],**stats(values),'difference':d})
        result['micro'][name]={'layers':record,'sum_medians':sum(r['median'] for r in record)};save()
        print('MICRO',name,result['micro'][name]['sum_medians'],max(r['difference']['max_abs'] for r in record),flush=True)
    if a.stage=='micro':
        result['passed']=True;save()
    else:
        from test_scan import run as run_units
        result['units']=run_units()
        selected=json.loads(Path('/work/selection.json').read_text())['variants']
        result['case_count']=len(cases);result['question_count']=sum(len(c['questions']) for c in cases)
        result['baseline_case_seconds']={}
        for case in cases:
            qwen.torch_chunk_gated_delta_rule=baseline
            t=time.perf_counter()
            result['baseline_cases'][case['id']]=invoke(case)
            result['baseline_case_seconds'][case['id']]=time.perf_counter()-t
        save()
        for name in selected:
            qwen.torch_chunk_gated_delta_rule=variants[name]
            record={'cases':{}}
            for case in cases:
                count_before=dict(getattr(variants[name],'counts',{}))
                t=time.perf_counter();response=invoke(case);elapsed=time.perf_counter()-t
                comp=compare(response,result['baseline_cases'][case['id']])
                count_after=getattr(variants[name],'counts',{})
                record['cases'][case['id']]={'seconds':elapsed,'comparison':comp,'response':response,
                    'scan_calls':{k:count_after[k]-v for k,v in count_before.items()}}
                print('CASE',name,case['id'],elapsed,comp,flush=True)
            record['passed']=all(r['comparison']['passed'] for r in record['cases'].values())
            record['scan_counts']=dict(getattr(variants[name],'counts',{}))
            result['model'][name]=record;save()
        acceptable=[name for name in selected if result['model'][name]['passed']]
        timings=[]
        for round_index in range(12):
            order=['baseline']+acceptable
            if round_index%2:order.reverse()
            for name in order:
                qwen.torch_chunk_gated_delta_rule=variants[name]
                invoke(single)
                state=operating_state()
                t=time.perf_counter();response=invoke(single);elapsed=time.perf_counter()-t
                comp=compare(response,result['baseline_cases']['single']);assert comp['passed']
                timings.append({'round':round_index,'variant':name,'seconds':elapsed,'comparison':comp,'operating_state_before':state})
                print('PAIR',round_index,name,elapsed,comp['max_probability_or_score_delta'],flush=True)
        result['paired']={'records':timings,'summary':{name:stats([r['seconds'] for r in timings if r['variant']==name]) for name in ['baseline']+acceptable}}
        save()
        if acceptable:
            best=min(acceptable,key=lambda n:result['paired']['summary'][n]['median']);qwen.torch_chunk_gated_delta_rule=variants[best]
            values=[];deltas=[]
            for i in range(100):
                t=time.perf_counter();response=invoke(single);values.append(time.perf_counter()-t)
                comp=compare(response,result['baseline_cases']['single']);assert comp['passed'];deltas.append(comp['max_probability_or_score_delta'])
                if (i+1)%10==0:print('STABILITY',best,i+1,values[-1],flush=True)
            result['selected_100']={'variant':best,**stats(values),'max_delta':max(deltas),'every_response_checked':True};save()
            budgets=[]
            for round_index in range(6):
                for name in (('baseline',best) if round_index%2==0 else (best,'baseline')):
                    scan_times=[]
                    def measured(*args,**kwargs):
                        t=time.perf_counter();output=variants[name](*args,**kwargs);scan_times.append(time.perf_counter()-t);return output
                    qwen.torch_chunk_gated_delta_rule=measured
                    t=time.perf_counter();response=invoke(single);elapsed=time.perf_counter()-t
                    assert len(scan_times)==6 and compare(response,result['baseline_cases']['single'])['passed']
                    budgets.append({'round':round_index,'variant':name,'total':elapsed,'scan':sum(scan_times),
                                    'other':elapsed-sum(scan_times),'six_scan_seconds':scan_times})
            result['budget']={'scope':'Light scan-boundary wall clock, includes scan input preparation and native call. Other is residual, not an exclusive operator profile.',
                              'records':budgets,'medians':{name:{k:statistics.median(r[k] for r in budgets if r['variant']==name) for k in ('total','scan','other')} for name in ('baseline',best)}};save()
        result['passed']=bool(acceptable);save()
finally:
    qwen.torch_chunk_gated_delta_rule=baseline
    restore_engine();engine.close()
    for callback in reversed(restores):callback()
