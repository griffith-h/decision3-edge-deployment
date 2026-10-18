"""Validate experiment record completeness and the actual measured graph path."""
from pathlib import Path
import hashlib,json,math,ast
P=Path(__file__).resolve().parent
load=lambda s:json.loads((P/s).read_text())
for name in ['row-unit','bucket-unit','planned-unit','bucket-primitive']:
 assert load('logs/'+name+'.json')['passed'],name
compiler=list((P/'logs').glob('compile-mlp-*-rows*.json'));assert len(compiler)==24
for p in compiler:
 d=json.loads(p.read_text());assert all(d[k]==0 for k in ['config_return','load_return','build_return','export_return']) and not d['quantized'],p
local,board=load('logs/local-artifacts.json'),load('logs/board-artifacts.json');assert len(local)==25
for k,v in local.items():assert board[k]==v,k
for k,v in board.items():
 p=P/k
 if p.is_file() and p.suffix in ['.py','.sh','.cpp']:assert hashlib.sha256(p.read_bytes()).hexdigest()==v,k
candidate=[]
for file,mode in [('screen.json','tile115'),('bucket-screen.json','bucket'),('planned-screen.json','planned')]:
 d=load('logs/'+file);assert d['passed'] and len(d['cases'])==13
 for key,c in d['cases'].items():
  records=c['variants'][mode]['records'];assert len(records)==(2 if key=='image' and mode!='tile115' else 4)
  candidate+=records
formal=load('logs/planned-formal.json');assert formal['passed'] and len(formal['cases'])==9
for c in formal['cases'].values():
 for mode in ['fixed','planned']:assert len(c['variants'][mode]['records'])==20
 candidate+=c['variants']['planned']['records']
assert len(formal['extra'])==72 and all(v['passed'] for v in formal['extra'])
candidate+=[v for v in formal['extra'] if v['mode']=='planned']
long=load('logs/long-screen.json');assert long['completed'] and len(long['cases'])==4
for c in long['cases'].values():
 for mode in ['fixed','planned']:assert len(c['variants'][mode]['records'])==2
 candidate+=c['variants']['planned']['records']
for v in candidate:
 assert v['optimized']==12 and v['fallback']==0 and v.get('range_fallback',0)==0 and not v.get('errors'),v
 assert v['graph_calls']>=12
 for k in ['max_delta','confidence_max_delta','all_numeric_max_delta']:assert math.isfinite(v[k]) and v[k]>=0
profile=load('logs/planned-profile.json');assert len(profile)==5
for modes in profile.values():
 for mode,d in modes.items():
  assert len(d['records'])==5
  for v in d['records']:
   assert len(v['layers'])==12 and v['seconds']>=v['mlp_seconds']
   if mode=='planned':assert v['counts']['optimized']==12 and v['counts']['fallback']==0
before,after=load('service-before.json'),load('service-after.json');assert after['passed'] and after['health']=='healthy' and not after['tests']
for k in ['id','started_at','config_hash','hostconfig_hash','files']:assert before[k]==after[k],k
assert len(after['responses'])==5 and all(v==before['responses'][0] for v in after['responses'])
r={'record_structure_passed':True,'compiled_graphs':len(compiler),'candidate_outputs':len(candidate),'all_probability_gates_passed':all(v['passed'] for v in candidate),'all_choices_unchanged':all(v['choices_unchanged'] for v in candidate),'all_12_mlp_layers_hit':True,'no_measured_runtime_or_range_fallbacks':True,'service_unchanged':True}
(P/'logs/record-validation.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
