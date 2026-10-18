"""Real mixed-shape/mode sequence, including both615 control banks."""
from accepted_owner import *
from pathlib import Path
p=Path('/experiment');refs=json.loads(Path('/fusion/logs/references.json').read_text());app=None;r={'records':[],'completed':False}
try:
 app=Owner();original_func=app.last_readout.original.__func__
 sequence=[('baseline','words-512'),('candidate','words-512'),('candidate','single'),('baseline','words-512'),('candidate','words-512'),('candidate','multi-wide'),('candidate','words-512')]
 for mode,key in sequence:
  app.use(mode);out=app.invoke(refs['cases'][key]['request']);rec={'mode':mode,'case':key,'output':out,'vs_current':detailed(out,refs['cases'][key]['output']),'handles':[len(app.engine.handles),len(app.native_mlp.handles)],**app.counts()};r['records'].append(rec);print('CACHE INTEGRATION',mode,key,rec['handles'],rec['fp32_contexts'],rec['cache_last_events'],flush=True)
  assert rec['vs_current']['passed'] and not rec['precise_errors'] and not rec['precise_range_fallbacks'] and not rec['projections']['errors'] and not rec['last_readout_errors']
  assert rec['native_mlp_calls']==(11 if mode=='candidate' and key=='words-512' else 0)
 assert any(e['bank']=='normal_fp32_mlp' and e['count']==33 for rec in r['records'] for e in rec['cache_last_events'])
 assert any(e['bank']=='half' and e['count']>=36 for rec in r['records'] for e in rec['cache_last_events']);r['completed']=True
finally:
 if app:
  app.close();assert app.model.logits.__func__ is original_func;app.close();assert app.model.logits.__func__ is original_func;r['double_close_restored_original_logits']=True;r['handles_after_close']=[len(app.engine.handles),len(app.native_mlp.handles),len(app.native_attention.handles)];assert r['handles_after_close']==[0,0,0]
 (p/'logs/cache-integration.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
