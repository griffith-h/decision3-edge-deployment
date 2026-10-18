from fused_owner import *
from pathlib import Path
from experiment import numeric
app=Owner();base=json.loads((P/'logs/screen.json').read_text());c=base['cases']['multi-wide']['request'];r={'request':c,'variants':{}}
try:
 app.use('current');ref=app.invoke(c);r['reference']=ref
 for mode in ['mlp_only','chunked','linear_only','fused','fused_guarded']:
  app.use(mode);before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(c);check=compare(out,ref);a,b=numeric(out['answers'],confidence=True),numeric(ref['answers'],confidence=True);worst=sorted([{'path':k,'reference':b[k],'candidate':v,'delta':abs(v-b[k])} for k,v in a.items()],key=lambda x:x['delta'],reverse=True)
  r['variants'][mode]={'seconds':time.perf_counter()-t,'output':out,**check,**app.counts(),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before,'worst_fields':worst[:8]};(P/'logs/decomposition.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print('DECOMPOSE',mode,r['variants'][mode]['seconds'],check,r['variants'][mode]['mlp_policy_fallbacks'],flush=True)
 # One additional natural-language B=2 input validates the batch guard away
 # from the synthetic ambiguous B=3 case that motivated it.
 extra=copy.deepcopy(base['cases']['chinese-0']['request']);extra['id']='wide2-natural';extra['state']=extra['state']*12;extra['questions']=dict(list(extra['questions'].items())[:2]);app.use('current');reference=app.invoke(extra);reg={'request':extra,'reference':reference,'records':[]}
 for repeat in range(2):
  app.use('fused_guarded');before=app.engine.cpu_fallbacks;t=time.perf_counter();actual=app.invoke(extra);reg['records'].append({'seconds':time.perf_counter()-t,**compare(actual,reference),**app.counts(),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before,'output':actual})
 (P/'logs/guarded-regression.json').write_text(json.dumps(reg,ensure_ascii=False,indent=2)+'\n');print('GUARDED NATURAL',reference['usage']['input_tokens'],[(v['seconds'],v['passed'],v['mlp_policy_fallbacks']) for v in reg['records']],flush=True)
 r['completed']=True;(P/'logs/decomposition.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
finally:app.close()
