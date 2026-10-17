"""Two persistent engines, alternating order; warmed trials, every output checked."""
import setup_helpers
import argparse,json,time,os,hashlib
from pathlib import Path
import torch
from common import load,compare,stats,state
from setup_helpers import configure
from adapter import install
from merged_adapter import PackedEngine
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['merge','balanced','baseline'],required=True);p.add_argument('--compile',default='none');p.add_argument('--rounds',type=int,default=12);a=p.parse_args()
snapshot='/models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e'
model,baseline,native,close=load(snapshot,safe_output=True);baseline.restore_forwards();baseline.restore_forwards=None
control={'mode':'none'};metadata={};cpu_restore=None
if a.compile!='none':
    from compiled_regions import install as install_cpu
    cpu_restore=install_cpu(model,control,metadata)
candidate=baseline if a.mode=='baseline' else PackedEngine(a.mode)
if candidate is not baseline:configure(candidate)
candidate.restore_forwards=None;current=None;cache={};modules=list(model.backbone.modules())
def retire():
    global current
    if current is None:return
    cache[id(current)]={id(m):m._bp_handle for m in modules if hasattr(m,'_bp_handle')}
    current.restore_forwards();current.restore_forwards=None;current=None
def use(engine,compile_mode):
    global current
    retire();control['mode']=compile_mode;engine.groups=[];engine.restore_forwards=install(model,engine)
    for m in modules:
        if hasattr(m,'_bp_handle'):m._bp_handle=cache.get(id(engine),{}).get(id(m))
    current=engine
body=json.loads(Path('/base/request.json').read_text())
def invoke():return model.system_one(state=body['state'],questions=body['questions'])
tag=a.mode+'-'+a.compile+('-static' if os.environ.get('D3_COMPILE_DYNAMIC')=='0' and a.compile!='none' else '');out=Path('/work/logs/paired-'+tag+'.json')
result={'tag':tag,'scope':'Same loaded model, two persistent warm engines (one if CPU only), alternating order, one untimed warmup before each timed trial; no HTTP. Different resource condition from single-engine100.','settings':state(),'records':[],
        'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').iterdir() if p.suffix in ('.py','.cpp','.so')}}
try:
    use(baseline,'none');invoke();reference=invoke()
    use(candidate,a.compile);invoke();assert compare(invoke(),reference)['passed']
    if os.environ.get('D3_COMPILE_DYNAMIC')=='0':control['freeze_shapes']=True
    for i in range(a.rounds):
        order=[('baseline',baseline,'none'),('candidate',candidate,a.compile)]
        if i%2:order.reverse()
        for name,engine,cpu in order:
            use(engine,cpu);torch.set_num_threads(3);invoke()
            t=time.perf_counter();response=invoke();elapsed=time.perf_counter()-t
            comp=compare(response,reference);assert comp['passed'],comp
            result['records'].append({'round':i,'variant':name,'seconds':elapsed,'comparison':comp})
            print('PAIR',tag,i,name,elapsed,flush=True)
    result.update(summary={n:stats([r['seconds'] for r in result['records'] if r['variant']==n]) for n in ('baseline','candidate')},
                  handles={'baseline':len(baseline.handles),'candidate':len(candidate.handles)},candidate_stats=candidate.stats,compile_stats=metadata,passed=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
finally:
    retire()
    if candidate is not baseline:candidate.close()
    if cpu_restore:cpu_restore()
    close()
