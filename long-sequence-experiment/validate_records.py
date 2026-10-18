from pathlib import Path
import json,hashlib,math
P=Path(__file__).resolve().parent
load=lambda name:json.loads((P/name).read_text())
unit=load('logs/unit.json');assert unit['passed'] and unit['noncausal_future_and_mask'] and unit['scan_state_crosses_512']
assert 'attention-scale-none' in unit['guards'] and 'attention-scale-nan' in unit['guards']
screen=load('logs/screen.json');assert screen['completed'] and len(screen['cases'])==3
for c in screen['cases'].values():
    assert c['handles_before']==c['handles_after']
    assert all(len(v['records'])==3 for v in c['variants'].values())
f=load('logs/formal.json');assert f['completed'] and f['native_paths_passed'] and f['candidate']=='both' and len(f['cases'])==9 and len(f['extra'])==24
counts={'single':(20,20),'words-410':(5,20),'words-512':(5,20),'words-921':(5,20),'natural-long':(3,10),'batch-long':(3,10),'boundary-512':(3,3),'multi-wide':(3,3),'image':(2,2)}
checked=[];current_failures=[];incremental_failures=[]
for key,c in f['cases'].items():
    a,b=c['variants']['baseline'],c['variants']['both'];assert (len(a['records']),len(b['records']))==counts[key]
    assert c['handles_before']==c['handles_after']
    assert c['latency_reduction']==1-b['timing']['median']/a['timing']['median']
    for mode,v in c['variants'].items():
        assert v['baseline_gate_passed']==all(r['vs_baseline']['passed'] for r in v['records'])
        assert v['current_gate_passed']==all(r['vs_current']['passed'] for r in v['records'])
        for r in v['records']:
            assert all(not x['errors'] and not x['native_fallback'] for x in r['long_kernels'].values())
            assert not r['projections']['errors']
            for ref in ('vs_baseline','vs_current'):
                m=r[ref];assert all(math.isfinite(m[k]) for k in ('max_delta','probability_max_delta','score_max_delta','noul_max_delta','confidence_max_delta'))
                assert m['passed']==(m['max_delta']<=.01 and m['choices_unchanged'])
            if mode=='both' and key in ('words-410','words-512','words-921','natural-long','batch-long'):
                assert all(x['optimized']==6 and x['guard_fallback']==0 for x in r['long_kernels'].values())
                assert r['projections']['optimized_projections']==42 and len(r['projections']['coverage'])==42
                assert r['projections']['native_max_rows']<=512 and not any(r['projections'][k] for k in ('range_fallback','runtime_fallback','guard_fallback'))
                assert r['engine_cpu_fallbacks']==0
    checked+=b['records']
    if not b['current_gate_passed']:current_failures.append(key)
    if not b['baseline_gate_passed']:incremental_failures.append(key)
    if key in ('single','boundary-512','multi-wide','image'):
        assert all(r['vs_baseline']['all_numeric_max_delta']==0 and r['vs_baseline']['choices_unchanged'] for r in b['records'])
        assert all(x['optimized']==0 for r in b['records'] for x in r['long_kernels'].values())
assert len(f['cases']['words-512']['profile'])==6
for r in f['extra']:assert r['vs_baseline']['all_numeric_max_delta']==0 and r['vs_baseline']['choices_unchanged']
checked+=f['extra']
before,after=load('service-before.json'),load('service-after.json');assert after['passed'] and after['running'] and after['health']=='healthy' and not after['tests']
for key in ('id','started_at','config_hash','hostconfig_hash','files'):assert before[key]==after[key],key
assert len(after['responses'])==5 and all(r==before['responses'][0] for r in after['responses'])
art=load('logs/artifacts.json');assert art['native_sources_match_frozen_origins'] and art['native_binaries_unchanged_since_measurement']
for name,digest in art['sources'].items():
    if not name.endswith('.so'):assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
result={'passed':True,'formal_candidate_outputs':len(checked),'long_native_paths_exercised':True,'native_errors':0,'handles_reused':True,'incremental_prediction_gate_failed_cases':incremental_failures,'current_prediction_gate_failed_cases':current_failures,'precision_failures_retained':True,'production_unchanged':True}
(P/'logs/validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
