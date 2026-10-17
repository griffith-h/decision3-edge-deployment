import setup_helpers
import argparse,json,time,hashlib,os
from pathlib import Path
import torch
from common import load,compare,stats,state
from setup_helpers import configure,cases
from adapter import install
from merged_adapter import PackedEngine
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['baseline','merge','balanced'],default='baseline');p.add_argument('--count',type=int,default=12);p.add_argument('--compile',default='none');a=p.parse_args()
snapshot='/models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e'
t=time.perf_counter();model,baseline,native,close=load(snapshot,safe_output=True)
engine=baseline;extra_close=None;cpu_restore=None;control={'mode':a.compile};compile_stats={}
if a.mode!='baseline' or a.compile!='none':
    baseline.restore_forwards();baseline.restore_forwards=None
    if a.compile!='none':
        from compiled_regions import install as install_cpu
        cpu_restore=install_cpu(model,control,compile_stats)
    if a.mode!='baseline':engine=PackedEngine(a.mode);configure(engine);extra_close=engine.close
    engine.groups=[];engine.restore_forwards=install(model,engine)
loaded=time.perf_counter()-t
all_cases=cases();single=next(c for c in all_cases if c['id']=='single')
refs=json.loads(Path('/remainder/logs/env-safe-default.json').read_text())['cases']
fp32refs=json.loads(Path('/remainder/logs/model-results.json').read_text())['baseline_cases']
def invoke(c):return model.system_one(**{k:c[k] for k in ('state','questions','images','videos') if k in c})
tag=a.mode+'-'+a.compile+('-static' if os.environ.get('D3_COMPILE_DYNAMIC')=='0' and a.compile!='none' else '');output=Path('/work/logs/'+tag+'.json')
result={'tag':tag,'settings':state(),'load_seconds':loaded,'case_count':len(all_cases),'question_count':sum(len(c['questions']) for c in all_cases),'cases':{},'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').glob('*.py')}}
def save():output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
try:
    t=time.perf_counter();invoke(single);result['cold_first_call_seconds']=time.perf_counter()-t
    invoke(single)
    if os.environ.get('D3_COMPILE_DYNAMIC')=='0':control['freeze_shapes']=True
    values=[];deltas=[];deltas_fp32=[]
    for i in range(a.count):
        t=time.perf_counter();response=invoke(single);values.append(time.perf_counter()-t)
        comp=compare(response,refs['single']['response']);assert comp['passed'],comp
        comp2=compare(response,fp32refs['single']);assert comp2['passed'],comp2
        deltas.append(comp['max_delta']);deltas_fp32.append(comp2['max_delta'])
        print('TIMED',tag,i+1,values[-1],flush=True)
    result.update(timing=stats(values),every_response_checked=True,max_timed_delta=max(deltas),max_timed_delta_vs_fp32=max(deltas_fp32));save()
    for c in all_cases:
        t=time.perf_counter();response=invoke(c);elapsed=time.perf_counter()-t
        comp=compare(response,refs[c['id']]['response']);comp2=compare(response,fp32refs[c['id']])
        result['cases'][c['id']]={'seconds':elapsed,'comparison':comp,'comparison_vs_fp32':comp2,'response':response}
        print('CASE',tag,c['id'],elapsed,comp,comp2,flush=True);save()
    result.update(passed=all(r['comparison']['passed'] and r['comparison_vs_fp32']['passed'] for r in result['cases'].values()),engine_stats=engine.stats,handles=len(engine.handles),layout=getattr(engine,'layout',[]),compile_stats=compile_stats,final_settings=state())
    if a.compile!='none':result['torch_compile_metrics']=torch._dynamo.utils.compile_times()
    save()
finally:
    if engine is not baseline and engine.restore_forwards:engine.restore_forwards();engine.restore_forwards=None
    if extra_close:extra_close()
    # Retire bridge before restoring CPU parent wrappers.
    if baseline.restore_forwards:baseline.restore_forwards();baseline.restore_forwards=None
    if cpu_restore:cpu_restore()
    close()
