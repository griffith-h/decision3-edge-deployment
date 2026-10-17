from application import Application,state,stats
from validate_candidate import sources
from choose import cpu_choice
import argparse,json,time,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['paired','single'],required=True);p.add_argument('--count',type=int,default=20);a=p.parse_args()
root=Path('/work');app=Application();source,selected,cfg=cpu_choice();configs={'baseline442':{'cpu_tune':False},'cpu-std':dict(cfg,cpu_tune=False),'cpu-a72':dict(cfg,cpu_tune=True)}
# Only a clearly faster and numerically qualified graph can join the formal comparison.
mc=root/'logs/mlp-multicore-screen.json'
if mc.exists():
 d=json.loads(mc.read_text());row=d.get('variants',{}).get('mlp-core3')
 if d.get('completed') and row and row['passed'] and row['timing']['median']<.97*d['variants']['cpu-a72']['timing']['median']:configs['mlp-core3']=row['config']
r={'stage':a.stage,'settings':state(),'sources':sources(),'configs':configs,'records':[],'scope':'fixed115-token single-question warm system_one, concurrency1, no HTTP; all values retained; .01 threshold fixed'}
path=root/('logs/formal-'+a.stage+'.json')
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
 if a.stage=='paired':
  names=list(configs)
  for name,c in configs.items():app.use(c);app.invoke();assert app.check(app.invoke())['passed']
  for i in range(a.count):
   order=names[i%len(names):]+names[:i%len(names)]
   for name in order:
    app.use(configs[name]);app.invoke();before=app.counts();t=time.perf_counter();out=app.invoke();elapsed=time.perf_counter()-t;check=app.check(out);assert check['passed'],(name,check)
    if name=='baseline442':assert check['vs_442']['max_delta']==0,'reference drift'
    counts=app.counts()
    if configs[name].get('mlp_graph'):assert counts['graphs']['optimized']-before['graphs']['optimized']==12 and not counts['graphs']['errors']
    elif configs[name].get('pipeline'):assert counts['cpu']['pipeline']['optimized']-before['cpu']['pipeline']['optimized']==12 and counts['cpu']['pipeline']['range_fallback']==0
    r['records'].append({'round':i,'variant':name,'seconds':elapsed,'after_state':state(),**check});print('FORMAL PAIR',i,name,elapsed,flush=True)
   save()
  r['timing']={n:stats([v['seconds'] for v in r['records'] if v['variant']==n]) for n in names};r['paired']={}
  for name in names[1:]:
   pairs=[{v['variant']:v['seconds'] for v in r['records'] if v['round']==i} for i in range(a.count)]
   r['paired'][name]={'round_reduction_median':statistics.median(1-v[name]/v['baseline442'] for v in pairs),'faster_rounds':sum(v[name]<v['baseline442'] for v in pairs)}
  cpus=['cpu-std','cpu-a72'];best=min(cpus,key=lambda n:r['timing'][n]['median'])
  if r['timing']['cpu-std']['median']<=r['timing'][best]['median']*1.01:best='cpu-std'
  if 'mlp-core3' in names and r['timing']['mlp-core3']['median']<r['timing'][best]['median']*.97:best='mlp-core3'
  r['selected']=best;r['selected_config']=configs[best];app.use(configs['baseline442']);r['baseline_after']=app.check(app.invoke());assert r['baseline_after']['vs_442']['max_delta']==0;r['passed']=True
 else:
  previous=json.loads((root/'logs/formal-paired.json').read_text());cfg=previous['selected_config'];r['selected']=previous['selected'];r['config']=cfg;app.use(cfg);app.invoke();app.invoke();times=[]
  for i in range(a.count):
   t=time.perf_counter();out=app.invoke();elapsed=time.perf_counter()-t;check=app.check(out);assert check['passed'],check
   r['records'].append({'index':i,'seconds':elapsed,'after_state':state(),**check});times.append(elapsed);print('FORMAL SINGLE',i,elapsed,flush=True)
  r['timing']=stats(times);r['counts_at_timing_end']=app.counts();r['handles_at_timing_end']=len(app.engine.handles);r['graphs_at_timing_end']=len(app.graphs);save()
  r.update(app.ev.validate());assert r['passed'];save();r['extra_repeats']=[]
  for repeat in range(3):
   for c in app.ev.extra:
    response=app.invoke(c);check=__import__('baseline').compare(response,app.ev.extra_refs[c['id']]);assert check['passed'],(repeat,c['id'],check);r['extra_repeats'].append({'repeat':repeat,'id':c['id'],'response':response,**check})
  r['passed']=True
 r['final_settings']=state();save()
finally:app.close()
