"""Measure temporary endpoint over LAN, leaving original port 8900 intact."""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('--base', default='http://192.168.1.164:8901')
p.add_argument('--count', type=int, default=20)
p.add_argument('--output', default='http-result.json')
p.add_argument('--ready-timeout', type=int, default=240)
args = p.parse_args()
base = args.base.rstrip('/')
body = json.loads((ROOT.parent/'request.json').read_text())
reference = json.loads((ROOT.parent/'rknn-matmul-experiment/logs/benchmark-all.json').read_text())['cases']['single']['cpu']['response']
def call(path, payload=None):
    request = urllib.request.Request(base+path, data=None if payload is None else json.dumps(payload, ensure_ascii=False).encode(),
                                     headers={'Content-Type':'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60 if payload else 3) as reply:
        result = json.load(reply)
    return result, time.perf_counter()-start
deadline = time.monotonic()+args.ready_timeout
while True:
    try:
        health, _ = call('/health')
        assert health['status'] == 'ready'
        break
    except (OSError, AssertionError):
        if time.monotonic()>deadline:
            raise
        time.sleep(2)
print('TEMPORARY_HTTP_READY', flush=True)
def numeric(x, path=''):
    found={}
    if isinstance(x,dict):
        for k,v in x.items():
            if k not in ('confidence','legend'):
                found.update(numeric(v,path+'/'+k))
    elif isinstance(x,(int,float)):
        found[path]=x
    return found
call('/v1/systemone', body)  # explicit HTTP warmup, excluded from timings
records=[]
for i in range(args.count):
    started = time.time()
    response, elapsed = call('/v1/systemone',body)
    actual, expected=numeric(response['answers']),numeric(reference['answers'])
    assert actual.keys()==expected.keys()
    delta=max(abs(actual[k]-expected[k]) for k in actual)
    assert delta<=0.01 and response['answers']['kind']['choice']==reference['answers']['kind']['choice']
    records.append({'seconds':elapsed,'max_delta_vs_historical_cpu':delta,
                    'started_at_unix':started,'ended_at_unix':started+elapsed})
    print('REQUEST',i+1,round(elapsed,6),flush=True)
try:
    call('/v1/systemone',{'questions':{}})
    raise AssertionError('Invalid request accepted')
except urllib.error.HTTPError as exc:
    assert exc.code==422
timings=[r['seconds'] for r in records]
result={'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'passed':True,'base_url':base,'health':health,'requests':records,'count':len(records),
        'median':statistics.median(timings),'mean':statistics.mean(timings),
        'p95_nearest_rank':sorted(timings)[math.ceil(0.95*len(timings))-1],
        'min':min(timings),'max':max(timings),'below_one_second':sum(t<1 for t in timings),
        'invalid_request_422':True,'scope':f'Warm single question, 115 tokens, concurrency 1; {args.count} LAN HTTP calls, no request/result memoization, excludes model startup.'}
(ROOT/'logs'/args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('requests','health')},ensure_ascii=False),flush=True)
