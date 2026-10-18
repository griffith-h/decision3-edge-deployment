from experiment import *
import copy,torch
app=Owner();allcases=json.loads((P/'http-cases.json').read_text());cases=allcases[:7];single=next(x for x in cases if x['id']=='single')
for words in [0,16,64,128,256,400]:
 c=copy.deepcopy(single);c.update(id=f'words-{words}',state=' '.join(['alpha']*words));cases.append(c)
r={'settings':state(),'scope':'warm per-case alternating fixed115 vs tiled115; no HTTP; first shape separately recorded','cases':{}};outpath=P/'logs/screen.json'
def save():outpath.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
 for c in cases:
  row={'variants':{},'request':c};r['cases'][c['id']]=row
  app.use('fixed');t=time.perf_counter();ref=app.invoke(c);row['first_fixed_seconds']=time.perf_counter()-t;row['reference']=ref;tokens=ref['usage']['input_tokens'];row['input_tokens']=tokens
  # Four rounds alternate order and preserve each raw duration.
  for mode in ('fixed','tile115'):row['variants'][mode]={'records':[]}
  for i in range(4):
   for mode in (('fixed','tile115') if i%2==0 else ('tile115','fixed')):
    app.use(mode);app.invoke(c);before=dict(app.bank.counts);t=time.perf_counter();out=app.invoke(c);elapsed=time.perf_counter()-t;check=compare(out,ref);assert check['passed'],(c['id'],mode,check)
    after=dict(app.bank.counts);v={'seconds':elapsed,**check,'graph_calls':after['graph_calls']-before['graph_calls'],'optimized':after['optimized']-before['optimized'],'fallback':after['fallback']-before['fallback'],'shapes':dict(after['shapes'])};row['variants'][mode]['records'].append(v)
  for mode,v in row['variants'].items():v['timing']=stats([x['seconds'] for x in v['records']]);v['passed']=all(x['passed'] for x in v['records'])
  row['reduction']=1-row['variants']['tile115']['timing']['median']/row['variants']['fixed']['timing']['median'];save();print('SCREEN',c['id'],tokens,[(m,round(v['timing']['median']*1000,2)) for m,v in row['variants'].items()],row['reduction'],flush=True)
 r['passed']=True;r['final_settings']=state();save()
finally:app.close()
