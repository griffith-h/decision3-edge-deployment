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
path=Path('/work/logs/qualified-paired.json')
def save():path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
try:
    full=json.loads(Path('/work/logs/final-validate-combined.json').read_text())
    core=json.loads(Path('/work/logs/final-validate-scan-core.json').read_text())
    assert full['variants']['scan']['passed']
    variants=['baseline','scan']
    if core['variants']['scan-core']['passed']:variants.append('scan-core')
    result['variant']='qualified-comparison';result['qualified_variants']=variants
    for name in variants:use(name);invoke();assert check(invoke())['passed']
    for i in range(a.count):
        order=variants[i%len(variants):]+variants[:i%len(variants)]
        for name in order:
            use(name);invoke();t=time.perf_counter();r=invoke();elapsed=time.perf_counter()-t;comp=check(r);assert comp['passed'],comp
            assert all(c['optimized']==12 and c['fallback']==0 for c in counts.values()),counts
            result['records'].append({'round':i,'variant':name,'seconds':elapsed,'after_state':state(),**comp});print('PAIR',i,name,elapsed,flush=True)
    result['timing']={n:stats([r['seconds'] for r in result['records'] if r['variant']==n]) for n in variants}
    result['paired']={}
    for name in variants[1:]:
        pairs=[{r['variant']:r['seconds'] for r in result['records'] if r['round']==i} for i in range(a.count)]
        result['paired'][name]={'round_reduction_median':statistics.median(1-r[name]/r['baseline'] for r in pairs),'faster_rounds':sum(r[name]<r['baseline'] for r in pairs)}
    selected='scan'
    if 'scan-core' in variants and result['timing']['scan-core']['median']<.99*result['timing']['scan']['median']:selected='scan-core'
    result['selected']=selected
    use('baseline');result['baseline_after']=check(invoke());assert result['baseline_after']['passed']
    result.update(passed=True,graph_contexts=len(pool),handles=len(engine.handles));save()
finally:
    retire()
    for g in pool.values():g.close()
    close()
