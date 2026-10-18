"""Recompute saved-output deltas and medians without importing model dependencies."""
from pathlib import Path
import json,statistics,math
P=Path(__file__).resolve().parent
load=lambda n:json.loads((P/n).read_text())
f=load('logs/formal.json');screen=load('logs/screen.json');replay=load('logs/replay.json');profile=load('logs/profile-selected.json')
assert f['completed'] and f['fresh_current_matches_prior'] and replay['completed'] and profile['completed']
assert load('logs/unit.json')['passed'] and load('logs/extra-unit.json')['passed'] and load('service-after.json')['passed']
assert load('logs/artifacts.json')['sources']
def flatten(x,path='',confidence=False):
    if isinstance(x,dict):return {p:n for k,v in x.items() if k!='legend' and (confidence or k!='confidence') for p,n in flatten(v,path+'/'+k,confidence).items()}
    if type(x) in (int,float):
        assert math.isfinite(x);return {path:x}
    return {}
def recompute(out,ref):
    a,b=flatten(out['answers']),flatten(ref['answers']);assert a.keys()==b.keys();delta={k:abs(v-b[k]) for k,v in a.items()}
    aa,bb=flatten(out['answers'],confidence=True),flatten(ref['answers'],confidence=True);assert aa.keys()==bb.keys()
    choice=all(v.get('choice')==ref['answers'][k].get('choice') for k,v in out['answers'].items());m=max(delta.values())
    return {'max_delta':m,'confidence_max_delta':max((abs(v-bb[k]) for k,v in aa.items() if k.endswith('/confidence')),default=0),'all_numeric_max_delta':max(abs(v-bb[k]) for k,v in aa.items()),'choices_unchanged':choice,'passed':choice and m<=.01,'max_prediction_field':max(delta,key=delta.get),'probability_max_delta':max((v for k,v in delta.items() if '/probabilities/' in k),default=0),'score_max_delta':max((v for k,v in delta.items() if k.endswith('/score')),default=0),'noul_max_delta':max((v for k,v in delta.items() if k.endswith('/noul')),default=0)}
def equality(a,b):
    assert a.keys()==b.keys()
    for k in a:
        if type(a[k]) is float:assert abs(a[k]-b[k])<1e-12,(k,a[k],b[k])
        else:assert a[k]==b[k],(k,a[k],b[k])
def stats_check(v):
    times=[r['seconds'] for r in v['records']];assert all(x>0 for x in times)
    assert abs(statistics.median(times)-v['timing']['median'])<1e-12
    assert abs(sorted(times)[math.ceil(.95*len(times))-1]-v['timing']['p95_nearest_rank'])<1e-12

def guards(rec):
    assert not rec['mlp_schedule']['errors'] and not rec['mlp']['errors'] and not rec['projections']['errors']
    assert rec['engine_cpu_fallbacks']==0 and rec['mlp']['range_fallback']==0
    assert all(not v['errors'] and not v['native_fallback'] for v in rec['long_kernels'].values())
    assert rec['projections']['native_max_rows']<=512

checks=0
for kind,count in [('attention',12),('scan',29)]:
    assert len(replay[kind])==count
    for v in replay[kind].values():
        assert len(v['records'])==18 and set(x['layer'] for x in v['records'])==set(range(6))
        assert all(sum(x['layer']==i for x in v['records'])==3 for i in range(6))
        t=sum(statistics.median(x['seconds'] for x in v['records'] if x['layer']==i) for i in range(6));assert abs(t-v['sum_layer_medians'])<1e-12
        assert v['passed']==all(x['passed'] for x in v['records']);checks+=18
for mode,v in screen['variants'].items():
    assert len(v['records'])==4;stats_check(v)
    for rec in v['records']:
        guards(rec)
        equality(recompute(rec['output'],screen['baseline_reference']),rec['vs_baseline'])
        equality(recompute(rec['output'],screen['current_reference']),rec['vs_current'])
for mode,v in f['pilot'].items():
    assert len(v['records'])==6;stats_check(v)
    for rec in v['records']:
        guards(rec)
        for kind in ('baseline','current'):equality(recompute(rec['output'],f['pilot_reference'][kind]),rec['vs_'+kind])
passing=[m for m in ('validate_mlp','combined') if f['pilot'][m]['passed']];winner=min(passing,key=lambda m:f['pilot'][m]['timing']['median'])
if winner=='combined' and 'validate_mlp' in passing and f['pilot']['combined']['timing']['median']>=.98*f['pilot']['validate_mlp']['timing']['median']:winner='validate_mlp'
assert winner==f['candidate']=='validate_mlp';assert load('selected.json')['scan']=={'kind':'original'}
expected={'words-512':20,'single':10,'words-410':5,'words-921':5,'natural-long':5,'batch-long':5,'boundary-512':3,'multi-wide':3,'image':2};assert f['cases'].keys()==expected.keys()
incfailed=[];curfailed=[];candidates=[]
for key,c in f['cases'].items():
    assert c['handles_before']==c['handles_after'] and c['max_rss_kib']>0
    for mode,v in c['variants'].items():
        assert len(v['records'])==expected[key];stats_check(v)
        for rec in v['records']:
            guards(rec)
            for kind in ('baseline','current'):equality(recompute(rec['output'],c[kind+'_reference']),rec['vs_'+kind])
            # Retained original attention + original scan math + identical MLP row plan.
            assert rec['vs_baseline']['all_numeric_max_delta']==0,(key,mode)
            if mode==winner:
                assert rec['mlp_schedule']['parallel_layers']==(12 if key=='words-512' else 0)
                if key=='words-512':
                    assert rec['mlp']['graph_calls']==36 and rec['mlp']['padded_rows']==300 and rec['projections']['optimized_projections']==42
                    assert rec['long_kernels']['attention']['optimized']==rec['long_kernels']['scan']['optimized']==6
        assert v['baseline_gate_passed']==all(x['vs_baseline']['passed'] for x in v['records'])
        assert v['current_gate_passed']==all(x['vs_current']['passed'] for x in v['records'])
    new=c['variants'][winner];candidates+=new['records']
    assert abs(c['reduction']-(1-new['timing']['median']/c['variants']['baseline']['timing']['median']))<1e-12
    if not new['baseline_gate_passed']:incfailed.append(key)
    if not new['current_gate_passed']:curfailed.append(key)
assert len(candidates)==58 and incfailed==[] and curfailed==['multi-wide']
assert len(f['extra'])==24 and len(set(r['id'] for r in f['extra']))==12
assert all(r['vs_baseline']['all_numeric_max_delta']==0 for r in f['extra'])
for rec in f['extra']:guards(rec)
assert len(profile['records'])==10 and sum(r['mode']==winner for r in profile['records'])==5
assert all(r['unwrapped']>=0 and r['vs_baseline']['all_numeric_max_delta']==0 for r in profile['records'])
r={'passed':True,'candidate':winner,'native_replay_records':checks,'formal_candidate_outputs':len(candidates),'extra_short_outputs':24,'pilot_records':18,'screen_records':20,'profile_records':10,'full_output_metrics_recomputed':True,'extra_short_current_reference_is_baseline_alias':True,'incremental_gate_failed_cases':incfailed,'current_gate_failed_cases':curfailed,'all_formal_incremental_numeric_deltas_zero':True,'production_preserved':True}
(P/'logs/validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
