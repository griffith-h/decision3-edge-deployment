"""Trace real large-shape Linear equality, plus public CPU BLAS thread state."""
import argparse,ctypes,json
from pathlib import Path
import sys
import torch
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
sys.path[:0]=['/release','/base','/work',a.model]
from d3_runtime import D3
from cpu_optimized_server import optimize
from rknn_linear import Engine as OriginalEngine
from cpu_attention_optimized import install_conv,install_scan,install_sdpa
from native_norm import install_norm,install_native_conv
from adapter import Engine,install
blas=None
for line in Path('/proc/self/maps').read_text().splitlines():
    if 'libopenblas' in line and '/' in line:
        lib=ctypes.CDLL(line.split()[-1])
        if hasattr(lib,'openblas_get_num_threads'):
            blas=lib.openblas_get_num_threads;blas.restype=ctypes.c_int
        break
control=ctypes.CDLL('/work/libcpu_control.so')
for name in ('cpu_fpcr','cpu_fpsr'):getattr(control,name).restype=ctypes.c_uint64
backend=lambda:{'torch':torch.get_num_threads(),'openblas':blas() if blas is not None else None,
                'omp_max':control.cpu_omp_max_threads(),'omp_limit':control.cpu_omp_thread_limit(),
                'fpcr':hex(control.cpu_fpcr()),'fpsr':hex(control.cpu_fpsr())}
torch.set_num_threads(3)
model=D3.from_pretrained(a.model,device='cpu',verify='full');model.model_name='vllm-sr/d3-edge';optimize(model)
stats={};restore_native=[install_conv(),install_scan(128),install_sdpa(),install_norm(model.backbone,stats),install_native_conv(stats)]
from rknn_linear import install as install_original,uninstall
import statistics,time
engines={'baseline':OriginalEngine(),'combined':Engine(fusion=True,parallel=True)}
cache={name:{} for name in engines}
modules=list(model.backbone.modules())
def activate(name):
    engine=engines[name]
    if name=='baseline':
        install_original(model,engine)
        for m in modules:
            if hasattr(m,'_rknn_handle'):m._rknn_handle=cache[name].get(id(m))
        def retire():
            for m in modules:
                if hasattr(m,'_rknn_handle'):cache[name][id(m)]=m._rknn_handle
            uninstall(model)
    else:
        engine.groups=[]
        restore=install(model,engine)
        for m in modules:
            if hasattr(m,'_bp_handle'):m._bp_handle=cache[name].get(id(m))
        def retire():
            for m in modules:
                if hasattr(m,'_bp_handle'):cache[name][id(m)]=m._bp_handle
            restore()
    return retire
cases=json.loads(Path('/references/request-cases.json').read_text())
single=next(c for c in cases if c['id']=='single');english=next(c for c in cases if c['id']=='english')
def invoke(case):return model.system_one(**{k:case[k] for k in ('state','questions','images','videos') if k in case})
def numeric(x,path=''):
    out={}
    if isinstance(x,dict):
        for k,v in x.items():
            if k not in ('confidence','legend'):out.update(numeric(v,path+'/'+k))
    elif isinstance(x,(int,float)):out[path]=x
    return out

def difference(x,y):
    actual,expected=numeric(x['answers']),numeric(y['answers']);assert actual.keys()==expected.keys()
    delta=max(abs(actual[k]-expected[k]) for k in actual)
    same=all(v.get('choice')==y['answers'][k].get('choice') for k,v in x['answers'].items())
    assert same and delta<=.01,(same,delta)
    return delta
references={};prime={};records=[]
try:
    for name in ('baseline','combined'):
        retire=activate(name)
        try:
            torch.set_num_threads(3)
            invoke(single)
            response=invoke(english)
            if name=='baseline':references['english']=response
            prime[name]={'english_delta':difference(response,references['english']),'backend':backend(),'response':response}
            response=invoke(single)
            if name=='baseline':references['single']=response
            assert difference(response,references['single'])<=.01
        finally:retire()
    for round_index in range(10):
        for name in (('baseline','combined') if round_index%2==0 else ('combined','baseline')):
            retire=activate(name)
            try:
                # A controlled per-trial process setting; no host settings or
                # deployed service changes. One explicit warmup before timing.
                torch.set_num_threads(3)
                invoke(single)
                before=backend();t=time.perf_counter();response=invoke(single);elapsed=time.perf_counter()-t
                row={'round':round_index,'variant':name,'seconds':elapsed,'backend_before':before,'backend_after':backend(),
                     'max_delta_vs_baseline':difference(response,references['single'])}
                records.append(row);print('PAIRED',round_index,name,elapsed,flush=True)
            finally:retire()
    summary={name:{'median':statistics.median(r['seconds'] for r in records if r['variant']==name),
                   'min':min(r['seconds'] for r in records if r['variant']==name),
                   'max':max(r['seconds'] for r in records if r['variant']==name)} for name in engines}
    speedups=[next(r['seconds'] for r in records if r['round']==i and r['variant']=='baseline')/next(r['seconds'] for r in records if r['round']==i and r['variant']=='combined') for i in range(10)]
    assert all(len(e.handles)==78 for e in engines.values())
    result={'passed':True,'scope':'Same loaded model and frozen CPU kernels; original baseline vs fusion+split-core candidate. Two persistent78-handle engines. Alternating trial order, Torch3 reset before each trial, explicit single warmup per trial. Each timed output checked. Not a deployed/HTTP benchmark.',
             'prime':prime,'records':records,'summary':summary,'median_paired_speedup':statistics.median(speedups),
             'candidate_faster_rounds':sum(v>1 for v in speedups),'engine_handle_counts':{k:len(v.handles) for k,v in engines.items()}}
    Path('/work/logs/paired-results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('PAIRED_COMPLETE',summary,result['median_paired_speedup'],flush=True)
finally:
    for engine in engines.values():engine.close()
    for callback in reversed(restore_native):callback()
