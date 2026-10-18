from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
load=lambda s:json.loads((P/s).read_text())
for name in ['unit','native-unit','extra-guards']:assert load('logs/'+name+'.json')['passed'],name
screen=load('logs/screen.json');assert screen['completed'] and len(screen['inventory']['selected'])==42 and len(screen['inventory']['groups'])==12
# The first unrestricted composition is deliberately NOT a passing result.
assert not screen['cases']['multi-wide']['variants']['chunked']['passed']
decomp=load('logs/decomposition.json');assert decomp['completed'];assert decomp['variants']['fused_guarded']['passed']
natural=load('logs/guarded-regression.json');assert len(natural['records'])==2 and all(x['passed'] and x['mlp_policy_fallbacks']==12 for x in natural['records'])
fast=load('logs/fast-batch.json');assert fast['completed'] and len(fast['records'])==20 and fast['probability_only_gate_passed'] and not fast['prediction_gate_passed'];assert fast['handles_after_warmup']==fast['handles_after_timing'];assert all(v['choices_unchanged'] and v['projections']['optimized_projections']==42 and v['engine_cpu_fallbacks']==0 and not v['projections']['errors'] for v in fast['records'])
f=load('logs/formal.json');assert f['completed'] and f['passed'];mode=f['candidate'];assert mode=='fused_guarded'
assert len(f['cases'])==4 and len(f['extra'])==24
checked=[];coverage=set(v['name'] for v in f['inventory']['selected'])
for key,c in f['cases'].items():
 a,b=c['variants']['mlp_only']['records'],c['variants'][mode]['records'];assert len(a)==(20 if key=='single' else 5);assert len(b)==(5 if key=='multi-wide' else 20)
 assert c['handles_after_warmup']==c['handles_after_timing']
 for r in b:
  assert r['passed'] and not r['projections']['errors']
  if key!='single':
   assert set(r['projections']['coverage'])==coverage and r['projections']['optimized_projections']==42
   assert 0<r['projections']['native_max_rows']<=512 and not any(r['projections'][k] for k in ['range_fallback','runtime_fallback','guard_fallback'])
   assert r['engine_cpu_fallbacks']==3*r['mlp_policy_fallbacks']
  if key=='multi-wide':assert r['mlp_policy_fallbacks']==12 and r['mlp']['fallback']==12
  else:assert r['mlp_policy_fallbacks']==0
 checked+=b
assert all(r['passed'] for r in f['extra']);checked+=f['extra']
for phase in ['before','after']:assert load('service-'+phase+'.json')['health']=='healthy'
before,after=load('service-before.json'),load('service-after.json');assert after['passed'] and not after['tests']
for key in ['id','started_at','config_hash','hostconfig_hash','files']:assert before[key]==after[key],key
assert all(r==before['responses'][0] for r in after['responses'])
artifacts=load('logs/artifacts.json')
for name,digest in artifacts['sources'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest
r={'passed':True,'formal_candidate_outputs':len(checked),'protected_probability_gates_passed':all(x['passed'] for x in checked),'selected_non_mlp_projections':42,'native_max_rows':512,'handles_reused':True,'fast_batch_samples':20,'fast_batch_score_gate_failure_retained':True,'failed_unprotected_route_retained':True,'production_unchanged':True}
(P/'logs/validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
