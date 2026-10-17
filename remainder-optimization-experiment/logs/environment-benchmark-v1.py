"""Process-local waiting policy test; no host/service settings changed."""
import argparse,copy,ctypes,json,os,resource,time
from pathlib import Path
from common import load,compare,stats,sources,state
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--tag',required=True)
p.add_argument('--count',type=int,default=30);p.add_argument('--cases',action='store_true');a=p.parse_args()
model,engine,native,close=load(a.model,half_output=True)
cases=json.loads(Path('/references/request-cases.json').read_text());single=next(c for c in cases if c['id']=='single')
image=json.loads(Path('/references/golden-reference.json').read_text())['image'];image['state']='Describe the attached image.';cases.append({'id':'image',**image})
for name,text in [('novel-math','请计算 37 乘以 19，并检查结果。'),('novel-chat','今天下班后想和朋友散步聊天，有什么轻松的话题？')]:
    item=copy.deepcopy(single);item.update(id=name,state=text);cases.append(item)
refs=json.loads(Path('/work/logs/model-results.json').read_text())['baseline_cases']
def invoke(c):return model.system_one(**{k:c[k] for k in ('state','questions','images','videos') if k in c})
result={'tag':a.tag,'settings':state(),'environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','OMP_WAIT_POLICY','GOMP_SPINCOUNT')},
        'sources':sources(),'scope':'Single FP16-output engine + native scan; one request at a time, warmed system_one, not HTTP. Process environment only.', 'cases':{}}
out=Path('/work/logs/env-'+a.tag+'.json')
try:
    invoke(single);invoke(single)
    values=[];comps=[];before=resource.getrusage(resource.RUSAGE_SELF)
    for i in range(a.count):
        t=time.perf_counter();response=invoke(single);values.append(time.perf_counter()-t)
        comp=compare(response,refs['single']);assert comp['passed'];comps.append(comp)
        if (i+1)%10==0:print('ENV',a.tag,i+1,values[-1],flush=True)
    after=resource.getrusage(resource.RUSAGE_SELF)
    result['timing']=stats(values);result['every_response_checked']=True;result['max_timed_delta']=max(c['max_delta'] for c in comps)
    result['cpu_usage']={'user_seconds':after.ru_utime-before.ru_utime,'system_seconds':after.ru_stime-before.ru_stime,
                         'cpu_seconds_per_request':((after.ru_utime-before.ru_utime)+(after.ru_stime-before.ru_stime))/a.count,
                         'voluntary_switches':after.ru_nvcsw-before.ru_nvcsw,'involuntary_switches':after.ru_nivcsw-before.ru_nivcsw}
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for case in (cases if a.cases else [single]):
        response=invoke(case);comp=compare(response,refs[case['id']])
        result['cases'][case['id']]={'comparison':comp,'response':response};assert comp['passed'],(case['id'],comp)
    result['final_settings']=state();result['passed']=True
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
finally:close()
