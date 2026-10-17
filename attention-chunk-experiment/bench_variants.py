from baseline import load_baseline,compare,stats,state,cases
import argparse,json,time,hashlib,traceback,torch
from pathlib import Path
from chunk_blas2 import install as install_scan
from attention_graph import install as install_graph
p=argparse.ArgumentParser();p.add_argument('--variant',default='all');p.add_argument('--count',type=int,default=16);a=p.parse_args()
model,engine,base_counts,close=load_baseline();restores=[];counts={}
allcases=cases();single=next(c for c in allcases if c['id']=='single')
refs=json.loads(Path('/previous/logs/final-single-conversion-conv-prepare.json').read_text())['cases'];fp32=json.loads(Path('/remainder/logs/model-results.json').read_text())['baseline_cases']
def invoke(c=single):return model.system_one(**{k:c[k] for k in ('state','questions','images','videos') if k in c})
def check(r,c=single):
    x=compare(r,refs[c['id']]['response']);y=compare(r,fp32[c['id']]);return {'vs_454':x,'vs_fp32':y,'passed':x['passed'] and y['passed']}
def retire():
    for restore in reversed(restores):restore()
    restores.clear()
def use(name):
    global counts
    retire();counts={}
    if name in ('scan','combined'):
        r,c=install_scan(32,64);restores.append(r);counts['scan']=c
    if name in ('graph','combined'):
        r,c=install_graph(model);restores.append(r);counts['graph']=c
    torch.set_num_threads(3)
result={'settings':state(),'variants':{},'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').iterdir() if p.suffix in ('.py','.c','.cpp','.so')}}
path=Path('/work/logs/variants-'+a.variant+'.json')
def save():path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
try:
    for name in (['baseline','scan','graph','combined'] if a.variant=='all' else [a.variant]):
        row={'cases':{},'records':[]};result['variants'][name]=row
        try:
            use(name);invoke();invoke();times=[]
            for i in range(a.count):
                t=time.perf_counter();r=invoke();times.append(time.perf_counter()-t);comp=check(r);row['records'].append({'seconds':times[-1],**comp});print('TIME',name,i,times[-1],comp,flush=True)
            row['timing']=stats(times)
            for c in allcases:
                before={k:dict(v) for k,v in counts.items()};r=invoke(c);after={k:dict(v) for k,v in counts.items()}
                row['cases'][c['id']]={'response':r,**check(r,c),'optimized_hits':{k:after[k]['optimized']-before[k]['optimized'] for k in before}}
                save()
            row['counts']=counts;row['passed']=all(r['passed'] for r in row['records']) and all(r['passed'] for r in row['cases'].values());save()
        except Exception:row['error']=traceback.format_exc();row['passed']=False;print(row['error'],flush=True);save()
    retire();result['baseline_after']=check(invoke());result['handles']=len(engine.handles);save()
finally:retire();close()
