from loader import open_model,compare,stats,state,PackedEngine,install
import ctypes,json,time,hashlib,traceback
from pathlib import Path
import torch
from setup_helpers import cases
from conv_btc import install as install_conv
from prepared_scan import install as install_prepare
model,baseline,native,close=open_model();baseline.restore_forwards();baseline.restore_forwards=None
cache_engine=PackedEngine('balanced');original=cache_engine.lib;cache_engine.lib=ctypes.CDLL('/work/libconversion_bridge.so')
for name in ('mm_error','mm_create_core','mm_run','mm_destroy','mm_prepare_bf16','mm_copy_prepared','mm_run_prepared_bf16','mm_run_bf16','mm_run_prepared_scatter_bf16'):
    src=getattr(original,name);dst=getattr(cache_engine.lib,name);dst.argtypes=src.argtypes;dst.restype=src.restype
cache_engine.restore_forwards=None
modules=list(model.backbone.modules());handle_cache={};current=None;cpu_restore=None;cpu_counts=None

def retire():
    global current,cpu_restore
    if current is not None:
        handle_cache[id(current)]={id(m):m._bp_handle for m in modules if hasattr(m,'_bp_handle')}
        current.restore_forwards();current.restore_forwards=None;current=None
    if cpu_restore:cpu_restore();cpu_restore=None

def use(engine,conv):
    global current,cpu_restore,cpu_counts
    retire()
    if conv:
        conv_restore,conv_counts=install_conv();cpu_restore=conv_restore;cpu_counts={'conv':conv_counts}
        if conv=='prepare':
            prepare_restore,prepare_counts=install_prepare()
            def restore_both():prepare_restore();conv_restore()
            cpu_restore=restore_both;cpu_counts['prepare']=prepare_counts
    else:cpu_counts={}
    engine.groups=[];engine.restore_forwards=install(model,engine)
    for m in modules:
        if hasattr(m,'_bp_handle'):m._bp_handle=handle_cache.get(id(engine),{}).get(id(m))
    current=engine;torch.set_num_threads(3)
allcases=cases();single=next(c for c in allcases if c['id']=='single')
refs=json.loads(Path('/projection/logs/scatter-balanced-none.json').read_text())['cases'];fp32=json.loads(Path('/remainder/logs/model-results.json').read_text())['baseline_cases']
def invoke(c):return model.system_one(**{k:c[k] for k in ('state','questions','images','videos') if k in c})
result={'settings':state(),'variants':{},'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').iterdir() if p.suffix in ('.py','.c','.cpp','.so')}}
def save():Path('/work/logs/screen2.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
try:
    for name,engine,conv in [('baseline',baseline,False),('conv-prepare',baseline,'prepare'),('conversion-conv-prepare',cache_engine,'prepare')]:
        row={'cases':{}};result['variants'][name]=row
        try:
            use(engine,conv);invoke(single);invoke(single);times=[]
            for i in range(16):
                t=time.perf_counter();r=invoke(single);times.append(time.perf_counter()-t)
                assert compare(r,refs['single']['response'])['passed'];assert compare(r,fp32['single'])['passed']
                print('TIMED',name,i,times[-1],flush=True)
            row['timing']=stats(times)
            for c in allcases:
                r=invoke(c);comp=compare(r,refs[c['id']]['response']);comp2=compare(r,fp32[c['id']]);row['cases'][c['id']]={'response':r,'comparison':comp,'comparison_vs_fp32':comp2}
                assert comp['passed'] and comp2['passed'],(c['id'],comp,comp2)
            row.update(passed=True,conv_counts=cpu_counts,engine_stats=dict(engine.stats),handles=len(engine.handles));save()
        except Exception:
            row.update(passed=False,error=traceback.format_exc());print(row['error'],flush=True);save()
    result['final_settings']=state();save()
finally:retire();cache_engine.close();close()
