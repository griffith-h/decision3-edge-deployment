"""Fresh outputs against original current and fifth-stage qualified outputs."""
from accepted_owner import *
from pathlib import Path
import resource
p=Path('/experiment');refs=json.loads(Path('/fusion/logs/references.json').read_text());old=json.loads(Path('/fusion/logs/final-regression.json').read_text());app=None;r={'cases':{},'completed':False};path=p/'logs/regression-native.json'
try:
 app=Owner();app.use('current');out=app.invoke(refs['cases']['words-512']['request']);assert out==refs['cases']['words-512']['output'];r['fresh_original_current_equal']=True
 for key,ref in refs['cases'].items():
  app.use('candidate');out=app.invoke(ref['request']);rec={'family':ref['family'],'request':ref['request'],'output':out,'vs_current':detailed(out,ref['output']),'vs_previous_qualified':detailed(out,old['cases'][key]['output']),**app.counts()};r['cases'][key]=rec;path.write_text(json.dumps(r,ensure_ascii=False,indent=2));print('REGRESSION',key,rec['vs_current'],rec['vs_previous_qualified']['all_numeric_max_delta'],flush=True)
  assert rec['native_mlp_calls']==(11 if ref['output']['usage']['input_tokens']==615 else 0)
  assert rec['vs_current']['passed'] and not rec['precise_errors'] and not rec['precise_range_fallbacks'] and not rec['last_readout_errors'] and not rec['mlp']['errors'] and not rec['projections']['errors']
 r['rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;r['completed']=True
finally:
 if app:app.close();r['handles_after_close']=[len(app.engine.handles),len(app.native_mlp.handles),len(app.native_attention.handles)];assert r['handles_after_close']==[0,0,0]
 path.write_text(json.dumps(r,ensure_ascii=False,indent=2))
