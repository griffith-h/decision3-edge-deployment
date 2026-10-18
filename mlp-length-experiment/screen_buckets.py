from experiment import *
from bucket_mlp import BucketBank
import copy
app=Owner();app.bank.close();app.bank=BucketBank(app.model)
allcases=json.loads((P/'http-cases.json').read_text());cases=[c for c in allcases[:7] if c['id']!='image'];single=next(x for x in cases if x['id']=='single')
for words in [0,16,64,128,256,400]:
 c=copy.deepcopy(single);c.update(id=f'words-{words}',state=' '.join(['alpha']*words));cases.append(c)
cases+=[next(x for x in allcases if x['id']=='image')]
r={'settings':state(),'scope':'4 alternating warmed repeats per mode, isolated model method; baseline is current fixed115 graph service equivalent','cases':{}};path=P/'logs/bucket-screen.json'
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
 for c in cases:
  row={'request':c,'variants':{m:{'records':[]} for m in ('fixed','bucket')}};r['cases'][c['id']]=row
  app.use('fixed');t=time.perf_counter();ref=app.invoke(c);row['reference']=ref;row['input_tokens']=ref['usage']['input_tokens'];row['first_fixed_seconds']=time.perf_counter()-t;save()
  # Image: two pairs suffice for first coverage, formal text lengths use more repeats later.
  for i in range(2 if c['id']=='image' else 4):
   for mode in (('fixed','bucket') if i%2==0 else ('bucket','fixed')):
    app.use(mode);t=time.perf_counter();warm=app.invoke(c);setup=time.perf_counter()-t
    app.bank.reset_counts();t=time.perf_counter();out=app.invoke(c);elapsed=time.perf_counter()-t;check=compare(out,ref)
    counts=copy.deepcopy(app.bank.counts);v={'seconds':elapsed,'warmup_seconds':setup,**check,**counts};row['variants'][mode]['records'].append(v);save()
    assert check['passed'],(c['id'],mode,check)
  for mode,v in row['variants'].items():v['timing']=stats([x['seconds'] for x in v['records']]);v['passed']=all(x['passed'] for x in v['records'])
  row['reduction']=1-row['variants']['bucket']['timing']['median']/row['variants']['fixed']['timing']['median'];save();print('BUCKET',c['id'],row['input_tokens'],[(m,round(v['timing']['median']*1000,2)) for m,v in row['variants'].items()],row['reduction'],flush=True)
 r['passed']=True;r['final_settings']=state();save()
finally:app.close()
