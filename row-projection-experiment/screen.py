from projected_owner import *
from pathlib import Path
app=Owner();cases=json.loads(Path('/mlp/http-cases.json').read_text());single=next(c for c in cases if c['id']=='single');outpath=P/'logs/screen.json'
r={'settings':state(),'inventory':app.projections.inventory,'calibration':app.bank.calibration,'cases':{}}
probe=[]
for words in [408,409,410,512]:
 c=copy.deepcopy(single);c.pop('baseline',None);c.update(id=f'words-{words}',state=' '.join(['alpha']*words));probe.append(c)
probe+=[cases[0],next(c for c in cases if c['id']=='single'),next(c for c in cases if c['id']=='image')]
# A genuine multi-question case crossing 512 flattened rows.
c=copy.deepcopy(cases[0]);c.update(id='multi-wide',state=' '.join(['alpha']*128));probe.insert(4,c)
def save():outpath.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
 for c in probe:
  print('BEGIN',c['id'],flush=True);row={'request':c,'variants':{m:{'records':[]} for m in ['current','mlp_only','single','chunked']}};r['cases'][c['id']]=row
  app.use('current');t=time.perf_counter();ref=app.invoke(c);row['reference']=ref;row['input_tokens']=ref['usage']['input_tokens'];row['first_current_seconds']=time.perf_counter()-t;save();print('REFERENCE',c['id'],row['input_tokens'],row['first_current_seconds'],flush=True)
  # Two pairs for initial feasibility; long original/MLP-only traces remain separate.
  for repeat in range(2):
   modes=['current','mlp_only','single','chunked'] if repeat==0 else ['chunked','single','mlp_only','current']
   for mode in modes:
    app.use(mode);t=time.perf_counter();out=app.invoke(c);elapsed=time.perf_counter()-t;record={'seconds':elapsed,**compare(out,ref),**app.counts()};row['variants'][mode]['records'].append(record);save();print('MEASURE',c['id'],mode,round(elapsed,4),record['max_delta'],record['confidence_max_delta'],record['passed'],record['projections']['optimized_projections'],flush=True)
  for mode,v in row['variants'].items():v['timing']=stats([x['seconds'] for x in v['records']]);v['passed']=all(x['passed'] for x in v['records'])
  save()
 r['completed']=True;r['passed']=all(c['variants']['chunked']['passed'] for c in r['cases'].values());r['final_settings']=state();save()
finally:app.close()
