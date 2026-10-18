"""Probe the pre-existing RKNN matmul row limit without imposing a new input limit."""
from experiment import *
from planned_mlp import PlannedBank
import copy
app=Owner();app.bank.close();app.bank=PlannedBank(app.model)
single=next(x for x in json.loads((P/'http-cases.json').read_text()) if x['id']=='single')
r={'settings':state(),'calibration':app.bank.calibration,'scope':'512 matmul row boundary, one warmup plus 2 alternating measured repeats per variant; not a percentile/SLA','cases':{}};path=P/'logs/long-screen.json'
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
 for words in [408,409,410,512]:
  c=copy.deepcopy(single);c.pop('baseline',None);c.update(id=f'words-{words}',state=' '.join(['alpha']*words));row={'request':c,'variants':{m:{'records':[]} for m in ['fixed','planned']}};r['cases'][c['id']]=row
  print('LONG BEGIN',c['id'],flush=True);app.use('fixed');t=time.perf_counter();ref=app.invoke(c);row['reference']=ref;row['input_tokens']=ref['usage']['input_tokens'];row['first_fixed_seconds']=time.perf_counter()-t;print('LONG FIRST FIXED',c['id'],row['input_tokens'],row['first_fixed_seconds'],flush=True)
  app.use('planned');t=time.perf_counter();out=app.invoke(c);row['first_planned_seconds']=time.perf_counter()-t;row['first_check']=compare(out,ref);save();print('LONG FIRST PLANNED',c['id'],row['first_planned_seconds'],row['first_check'],flush=True)
  for repeat in range(2):
   for mode in (('fixed','planned') if repeat%2==0 else ('planned','fixed')):
    app.use(mode);t=time.perf_counter();out=app.invoke(c);elapsed=time.perf_counter()-t;check=compare(out,ref);record={'seconds':elapsed,**check,**copy.deepcopy(app.bank.counts)};row['variants'][mode]['records'].append(record);save()
  for mode,v in row['variants'].items():v['timing']=stats([x['seconds'] for x in v['records']]);v['passed']=all(x['passed'] and not x['errors'] for x in v['records'])
  row['reduction']=1-row['variants']['planned']['timing']['median']/row['variants']['fixed']['timing']['median'];save();print('LONG',c['id'],row['input_tokens'],[(m,round(v['timing']['median']*1000,2)) for m,v in row['variants'].items()],row['reduction'],row['variants']['planned']['passed'],flush=True)
 r['completed']=True;r['passed']=all(v['variants']['planned']['passed'] and v['first_check']['passed'] for v in r['cases'].values());r['final_settings']=state();save()
finally:app.close()
