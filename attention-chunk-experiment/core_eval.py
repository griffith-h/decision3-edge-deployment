from baseline import load_baseline,compare,stats,state,cases
from chunk_blas2 import install as install_scan
from attention_graph import install as install_graph
from graph import Graph
from core_graph import Graph as CoreGraph
from attention_core import install as install_core
from target_shape_cases import build
import argparse,json,time,hashlib,statistics,torch
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validate','paired','single'],required=True);p.add_argument('--variant',default='scan-core',choices=['scan','core','scan-core']);p.add_argument('--count',type=int,default=20);a=p.parse_args()
model,engine,base_counts,close=load_baseline();restores=[];counts={};pool={}
# Borrowed graph views never destroy the persistent owner contexts during toggles.
class Borrowed:
    def __init__(self,path):
        key=str(path)
        if key not in pool:pool[key]=(CoreGraph(path) if path.name=='attention-core.rknn' else Graph(path))
        self.g=pool[key]
    def run(self,inputs):return self.g.run(inputs)
    def close(self):pass

def retire():
    for r in reversed(restores):r()
    restores.clear()
def use(name):
    global counts
    retire();counts={}
    if name in ('scan','scan-core'):
        r,c=install_scan(32,64);restores.append(r);counts['scan']=c
    if name in ('core','scan-core'):
        r,c=install_core(model,Borrowed);restores.append(r);counts['core']=c
    torch.set_num_threads(3)
allcases=cases();single=next(c for c in allcases if c['id']=='single')
refs=json.loads(Path('/previous/logs/final-single-conversion-conv-prepare.json').read_text())['cases'];fp32=json.loads(Path('/remainder/logs/model-results.json').read_text())['baseline_cases']
def invoke(c=single):return model.system_one(**{k:c[k] for k in ('state','questions','images','videos') if k in c})
def check(r,c=single):
    x=compare(r,refs[c['id']]['response']);y=compare(r,fp32[c['id']]);return {'vs_454':x,'vs_fp32':y,'passed':x['passed'] and y['passed']}
result={'stage':a.stage,'variant':a.variant,'settings':state(),'records':[],'cases':{},'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').iterdir() if p.suffix in ('.py','.c','.cpp','.so','.rknn')}}
path=Path('/work/logs/final-'+a.stage+'-'+a.variant+'.json')
def save():path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
try:
    if a.stage=='validate':
        extra=build(model,single);result['target_shape_cases']=extra;result['variants']={}
        use('baseline');extra_refs={c['id']:invoke(c) for c in extra};result['extra_baseline_responses']=extra_refs
        for name in ('core','scan-core'):
            use(name);row={'cases':{},'extra':{}};result['variants'][name]=row
            for c in allcases:
                before={k:v['optimized'] for k,v in counts.items()};r=invoke(c)
                row['cases'][c['id']]={'response':r,**check(r,c),'hits':{k:v['optimized']-before[k] for k,v in counts.items()}}
            for c in extra:
                before={k:v['optimized'] for k,v in counts.items()};r=invoke(c);comp=compare(r,extra_refs[c['id']])
                hits={k:v['optimized']-before[k] for k,v in counts.items()};assert all(v==6 for v in hits.values()),hits
                row['extra'][c['id']]={'response':r,'comparison':comp,'hits':hits};print('EXTRA',name,c['id'],comp,flush=True)
            row['counts']=counts;row['passed']=all(v['passed'] for v in row['cases'].values()) and all(v['comparison']['passed'] for v in row['extra'].values());save()
        result['passed']=all(r['passed'] for r in result['variants'].values());save()
    elif a.stage=='paired':
        for name in ('baseline',a.variant):use(name);invoke();assert check(invoke())['passed']
        for i in range(a.count):
            for name in (['baseline',a.variant] if i%2==0 else [a.variant,'baseline']):
                use(name);invoke();t=time.perf_counter();r=invoke();elapsed=time.perf_counter()-t;comp=check(r);assert comp['passed'],comp
                assert all(c['optimized']==12 and c['fallback']==0 for c in counts.values()),counts
                result['records'].append({'round':i,'variant':name,'seconds':elapsed,'after_state':state(),**comp});print('PAIR',i,name,elapsed,flush=True)
        result['timing']={n:stats([r['seconds'] for r in result['records'] if r['variant']==n]) for n in ('baseline',a.variant)}
        pairs=[{r['variant']:r['seconds'] for r in result['records'] if r['round']==i} for i in range(a.count)]
        result['round_reduction_median']=statistics.median(1-r[a.variant]/r['baseline'] for r in pairs);result['faster_rounds']=sum(r[a.variant]<r['baseline'] for r in pairs)
        use('baseline');result['baseline_after']=check(invoke());result['passed']=result['baseline_after']['passed'];save()
    else:
        use(a.variant);invoke();invoke();times=[]
        for i in range(a.count):
            t=time.perf_counter();r=invoke();times.append(time.perf_counter()-t);comp=check(r);assert comp['passed'],comp
            result['records'].append({'index':i,'seconds':times[-1],'after_state':state(),**comp});print('SINGLE',i,times[-1],flush=True)
        result['timing']=stats(times);result['timed_handles']=len(engine.handles);result['counts_at_timing_end']={k:dict(v) for k,v in counts.items()};save()
        for c in allcases:
            before={k:v['optimized'] for k,v in counts.items()};r=invoke(c);comp=check(r,c);assert comp['passed'],(c['id'],comp)
            result['cases'][c['id']]={'response':r,**comp,'hits':{k:v['optimized']-before[k] for k,v in counts.items()}};save()
        result['passed']=True;result['engine_stats']=engine.stats;save()
    result.update(graph_contexts=len(pool),handles=len(engine.handles),final_settings=state());save()
finally:
    retire()
    for g in pool.values():g.close()
    close()
