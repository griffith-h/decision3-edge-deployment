"""Verify original service preservation and the separate RKNN HTTP endpoint."""
import concurrent.futures
import json
import statistics
import time
import urllib.request
import urllib.error
from pathlib import Path

root=Path(__file__).resolve().parent
cpu='http://192.168.1.164:8899'
npu='http://192.168.1.164:8900'
def call(base,path,body=None,timeout=180):
    req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),
                               headers={'Content-Type':'application/json'})
    start=time.perf_counter()
    with urllib.request.urlopen(req,timeout=timeout) as response:
        value=json.load(response)
    return value,time.perf_counter()-start
deadline=time.monotonic()+150
while True:
    try:
        health,_=call(npu,'/health',timeout=5)
        assert health['status']=='ready'
        break
    except (OSError,AssertionError):
        if time.monotonic()>=deadline: raise
        time.sleep(2)
report={'cpu_health':call(cpu,'/health')[0],'npu_health':health,
        'npu_models':call(npu,'/v1/models')[0],'requests':[]}
try:
    call(npu,'/v1/systemone',{'questions':{}})
    raise AssertionError('Invalid request accepted')
except urllib.error.HTTPError as exc:
    assert exc.code==422
    report['invalid_request_returns_422']=True
single=json.loads((root.parent/'request.json').read_text())
response,seconds=call(cpu,'/v1/systemone',single)
expected=json.loads((root.parent/'rkllm-experiment/logs/cpu-preservation-check.json').read_text())['response']
assert response==expected, 'Existing CPU response changed'
report['cpu_preservation']={'passed':True,'seconds':seconds,'response':response}
bench=json.loads((root/'logs/benchmark-all.json').read_text())
cases=json.loads((root.parent/'rkllm-experiment/request-cases.json').read_text())
single_case=next(x for x in cases if x['id']=='single')
def numeric(value,path=''):
    result={}
    if isinstance(value,dict):
        for key,child in value.items():
            if key not in ('confidence','legend'):
                result.update(numeric(child,path+'/'+key))
    elif isinstance(value,(int,float)):
        result[path]=value
    return result
def verify(case):
    body={k:case[k] for k in ['state','questions','images','videos'] if k in case}
    body['model']='vllm-sr/d3-edge'
    response,seconds=call(npu,'/v1/systemone',body)
    reference=bench['cases'][case['id']]['npu']['response']
    baseline=bench['cases'][case['id']]['cpu']['response']
    assert response['model']==reference['model'] and response['usage']==reference['usage']
    actual=numeric(response['answers'])
    expected=numeric(reference['answers'])
    cpu_expected=numeric(baseline['answers'])
    assert set(actual)==set(expected)==set(cpu_expected)
    delta=max(abs(actual[k]-expected[k]) for k in actual)
    cpu_delta=max(abs(actual[k]-cpu_expected[k]) for k in actual)
    choices=all(v.get('choice')==baseline['answers'][key].get('choice') for key,v in response['answers'].items())
    assert delta<=0.01 and cpu_delta<=0.01 and choices, (case['id'],delta,cpu_delta)
    item={'case':case['id'],'seconds':seconds,'matches_benchmark_within_tolerance':True,
          'max_delta_vs_benchmark':delta,'max_delta_vs_cpu':cpu_delta,'choices_unchanged':choices,'response':response}
    print(json.dumps({k:v for k,v in item.items() if k!='response'}),flush=True)
    return item
for _ in range(3):
    report['requests'].append(verify(single_case))
report['single_npu_seconds_median']=statistics.median(x['seconds'] for x in report['requests'])
report['single_http_speedup']=seconds/report['single_npu_seconds_median']
report['requests'].append(verify(next(x for x in cases if x['id']=='chinese-0')))
image=json.loads((root.parent/'golden-reference.json').read_text())['image']
image['state']='Describe the attached image.'
report['requests'].append(verify({'id':'image',**image}))
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    report['concurrent_requests']=list(pool.map(verify,[single_case,next(x for x in cases if x['id']=='chinese-0')]))
report['passed']=True
report['max_probability_or_score_delta_vs_cpu']=max(x['max_delta_vs_cpu'] for x in report['requests']+report['concurrent_requests'])
(root/'logs/api-verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print('PASS: CPU unchanged; NPU HTTP probability delta <= 0.01; concurrent shape-changing requests agree.',flush=True)
