"""Recompute final metrics from raw outputs, without loading model/Torch."""
from pathlib import Path
import json,math,statistics
P=Path(__file__).resolve().parent
load=lambda name:json.loads((P/name).read_text())
def stats(values):
 values=sorted(values);assert values and all(math.isfinite(v) and v>0 for v in values)
 return {'count':len(values),'median':statistics.median(values),'min':min(values),'max':max(values),'p95_nearest_rank':values[math.ceil(.95*len(values))-1]}
def numbers(value,path='',confidence=True):
 if isinstance(value,dict):return {a:b for k,v in value.items() if k!='legend' and (confidence or k!='confidence') for a,b in numbers(v,path+'/'+k,confidence).items()}
 if isinstance(value,list):return {a:b for k,v in enumerate(value) for a,b in numbers(v,path+'/'+str(k),confidence).items()}
 if type(value) in (float,int):assert math.isfinite(value),(path,value);return {path:value}
 return {}
def compare(out,ref):
 assert out['usage']==ref['usage'] and out['model']==ref['model']
 a,b=numbers(out['answers'],confidence=False),numbers(ref['answers'],confidence=False);assert a.keys()==b.keys();d={k:abs(v-b[k]) for k,v in a.items()}
 ac,bc=numbers(out['answers']),numbers(ref['answers']);assert ac.keys()==bc.keys()
 same=out['answers'].keys()==ref['answers'].keys() and all(v.get('choice')==ref['answers'][k].get('choice') for k,v in out['answers'].items())
 confidence=max((abs(v-bc[k]) for k,v in ac.items() if k.endswith('/confidence')),default=0)
 return {'max_delta':max(d.values(),default=0),'probability_max_delta':max((v for k,v in d.items() if '/probabilities/' in k),default=0),'score_max_delta':max((v for k,v in d.items() if k.endswith('/score')),default=0),'noul_max_delta':max((v for k,v in d.items() if k.endswith('/noul')),default=0),'confidence_max_delta':confidence,'all_numeric_max_delta':max((abs(v-bc[k]) for k,v in ac.items()),default=0),'choices_unchanged':same,'passed':same and max(d.values(),default=0)<=.01}
def checked(out,ref,saved):
 result=compare(out,ref)
 for key,val in result.items():
  if type(val) in (int,float):assert abs(val-saved[key])<1.e-12,(key,val,saved[key])
  else:assert val==saved[key],key
 return result

def validate():
 refs=json.loads((P.parent/'fusion-pipeline-experiment/logs/references.json').read_text());prior=json.loads((P.parent/'fusion-pipeline-experiment/logs/final-regression.json').read_text());reg=load('logs/regression-native.json');formal=[load(f'logs/formal-{i}.json') for i in (1,2,3)];short=load('logs/short-formal.json');profile=load('logs/profile-native.json')
 assert reg['completed'] and reg['fresh_original_current_equal'] and len(reg['cases'])==34 and all(r['completed'] for r in formal+[short,profile])
 checks=[]
 for key,rec in reg['cases'].items():
  checks.append(checked(rec['output'],refs['cases'][key]['output'],rec['vs_current']));checked(rec['output'],prior['cases'][key]['output'],rec['vs_previous_qualified']);assert checks[-1]['passed']
  assert rec['native_mlp_calls']==(11 if refs['cases'][key]['output']['usage']['input_tokens']==615 else 0)
 assert reg['handles_after_close']==[0,0,0]
 assert len({r['run_id'] for r in formal})==3 and len({r['container_hostname'] for r in formal})==3
 for r in formal+[short]:
  assert r['handles_before']==r['handles_after'] and r['fds_before']==r['fds_after'] and r['handles_after_close']==[0,0,0]
  assert len(r['records'])==r['pairs']*2
  assert r['baseline_vs_fifth_regression']['choices_unchanged'] and r['baseline_vs_fifth_regression']['all_numeric_max_delta']<=1.e-6
  key='single' if r is short else 'words-512'
  for rec in r['records']:
   assert checked(rec['output'],refs['cases'][key]['output'],rec['vs_current'])['passed']
   assert checked(rec['output'],r['baseline_reference'],rec['vs_baseline'])['passed']
   assert not rec['engine_cpu_fallbacks'] and not rec['precise_errors'] and not rec['precise_range_fallbacks'] and not rec['last_readout_errors'] and not rec['mlp']['errors'] and not rec['projections']['errors']
   assert rec['native_mlp_calls']==(0 if r is short or rec['mode']=='baseline' else 11)
  for mode in ('baseline','candidate'):
   expected=stats([x['seconds'] for x in r['records'] if x['mode']==mode])
   for k,v in expected.items():assert abs(r['timing'][mode][k]-v)<1.e-12
 for rec in profile['records']:
  assert checked(rec['output'],refs['cases']['words-512']['output'],rec['vs_current'])['passed']
  assert rec['unwrapped']>=0 and all(x['end']>=x['start'] for x in rec['worker_intervals'])
 assert profile['handles_after_close']==[0,0,0]
 lifecycle=load('logs/bank-lifecycle.json');assert lifecycle['no_per_bank_fd_growth'] and lifecycle['owner_fds']['owned_handles_after_close']==[0,0,0]
 for name in ('stream-unit','boundary-unit','native-mlp-unit','split-mlp-unit','native-guards','split-guards','stream-guards','native-ranges','cache-unit','cache-integration','bank-lifecycle'):
  unit=load(f'logs/{name}.json');assert unit['completed'],name
 for name in ('screen-g128-t3','screen-g256-t3','screen-g512-t3','screen-cross-g256-t3','screen-native-projections','screen-native-stream-g256-t3','screen-split','screen-split-stream-g256-t3'):
  r=load(f'logs/{name}.json');assert r['completed']
  for rec in r['records']:assert checked(rec['output'],refs['cases']['words-512']['output'],rec['vs_current'])['passed']
 assets=load('logs/artifacts-after.json');service=load('service-after.json');assert assets['unchanged'] and service['passed'] and service['health']=='healthy' and not service['tests']
 values={m:[rec['seconds'] for r in formal for rec in r['records'] if rec['mode']==m] for m in ('baseline','candidate')};b,c=stats(values['baseline']),stats(values['candidate'])
 pairs=[{x['mode']:x['seconds'] for x in r['records'] if x['pair']==i} for r in formal for i in range(r['pairs'])]
 maxima={k:max(x[k] for x in checks) for k in ('probability_max_delta','score_max_delta','noul_max_delta','confidence_max_delta')}
 result={'passed':True,'goal_met':c['median']<1 and c['p95_nearest_rank']<1,'baseline':b,'candidate':c,'reduction':1-c['median']/b['median'],'pairs':len(pairs),'paired_candidate_wins':sum(x['candidate']<x['baseline'] for x in pairs),'paired_reduction_median':statistics.median(1-x['candidate']/x['baseline'] for x in pairs),'regression_cases':34,'regression_maxima':maxima,'max_delta_vs_previous_qualified':max(x['vs_previous_qualified']['all_numeric_max_delta'] for x in reg['cases'].values()),'formal_numeric_outputs':sum(len(x['records']) for x in formal),'short_numeric_outputs':len(short['records']),'profile_numeric_outputs':len(profile['records']),'formal_rss_max_kib':max(x['rss_kib'] for x in formal),'production_files_checked':len(service['files']),'artifact_files_checked':len(assets['files']),'selection':load('selection.json'),'bank_lifecycle_fd_counts':[len(x['fds_after_close']) for x in lifecycle['cycles']],'runtime_fds_after_pool':lifecycle['after_pool_shutdown'],'timing_exclusions':['split-mlp-initial-cold-control.json: incompletely warmed new down context, retained; entire protocol rerun'],'scope':'34 fixed regression requests, no business-accuracy, concurrency, cold-start or HTTP SLA claim'}
 (P/'logs/summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':validate()
