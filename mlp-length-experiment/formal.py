"""Alternating paired warmed timing and full-answer parity against current service equivalent."""
from experiment import *
from bucket_mlp import BucketBank
import copy
app=Owner();app.bank.close();app.bank=BucketBank(app.model)
screen=json.loads((P/'logs/bucket-screen.json').read_text());ids=['single','novel-math','words-16','words-64','words-128','chinese-0','words-256','words-400']
r={'settings':state(),'scope':'20 paired warm repetitions per text length; no HTTP, no concurrency, compare full answers per invocation','cases':{},'extra':[]};path=P/'logs/bucket-formal.json'
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
 for key in ids:
  c=screen['cases'][key]['request'];row={'request':c,'variants':{m:{'records':[]} for m in ['fixed','bucket']}};r['cases'][key]=row
  app.use('fixed');ref=app.invoke(c);row['reference']=ref;row['input_tokens']=ref['usage']['input_tokens']
  for mode in ['fixed','bucket']:app.use(mode);app.invoke(c)
  for repeat in range(20):
   for mode in (('fixed','bucket') if repeat%2==0 else ('bucket','fixed')):
    app.use(mode);t=time.perf_counter();out=app.invoke(c);elapsed=time.perf_counter()-t;check=compare(out,ref);counts=copy.deepcopy(app.bank.counts)
    record={'seconds':elapsed,**check,**counts};row['variants'][mode]['records'].append(record);save();assert check['passed'] and not counts['errors'],(key,mode,record)
  for mode,v in row['variants'].items():v['timing']=stats([x['seconds'] for x in v['records']])
  row['reduction']=1-row['variants']['bucket']['timing']['median']/row['variants']['fixed']['timing']['median'];save();print('FORMAL',key,row['input_tokens'],[(m,round(v['timing']['median']*1000,2)) for m,v in row['variants'].items()],row['reduction'],flush=True)
 # Additional independent existing 115-token prompts, three paired repetitions.
 for c in json.loads((P/'http-cases.json').read_text())[7:]:
  app.use('fixed');ref=app.invoke(c)
  for _ in range(3):
   for mode in ['fixed','bucket']:
    app.use(mode);out=app.invoke(c);check=compare(out,ref);r['extra'].append({'id':c['id'],'mode':mode,'input_tokens':out['usage']['input_tokens'],**check,**copy.deepcopy(app.bank.counts)});assert check['passed'] and not app.bank.counts['errors']
  save()
 r['passed']=True;r['final_settings']=state();save()
finally:app.close()
