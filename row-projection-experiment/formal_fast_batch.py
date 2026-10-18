from fused_owner import *
from metrics import detailed
app=Owner();source=json.loads((P/'logs/formal.json').read_text());c=source['cases']['multi-wide']['request'];ref=source['cases']['multi-wide']['reference'];r={'settings':state(),'request':c,'reference':ref,'reference_source':'formal.json same frozen snapshot, current path; evaluator reference only, candidate inference is never cached','records':[]}
try:
 app.use('fused');app.invoke(c);r['handles_after_warmup']=len(app.engine.handles)
 for i in range(20):
  app.use('fused');before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(c);rec={'seconds':time.perf_counter()-t,**detailed(out,ref),**app.counts(),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before};r['records'].append(rec)
  assert rec['choices_unchanged'] and not rec['projections']['errors'] and rec['engine_cpu_fallbacks']==0 and rec['projections']['optimized_projections']==42
  (P/'logs/fast-batch.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
 r['handles_after_timing']=len(app.engine.handles);assert r['handles_after_warmup']==r['handles_after_timing'];r['timing']=stats([x['seconds'] for x in r['records']]);r['completed']=True;r['prediction_gate_passed']=all(x['passed'] for x in r['records']);r['probability_only_gate_passed']=all(x['probability_max_delta']<=.01 for x in r['records']);r['final_settings']=state();(P/'logs/fast-batch.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print('FAST BATCH',r['timing']['median'],r['prediction_gate_passed'],r['probability_only_gate_passed'],flush=True)
finally:app.close()
