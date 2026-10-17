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
engine=Engine(fusion=True);original=OriginalEngine();restore_adapter=install(model,engine)
execute=engine.execute;names={id(m):n for n,m in model.backbone.named_modules()};handles={};captured=[]
def shadow(module,x,**kwargs):
    initial=backend()
    if id(module) not in handles:handles[id(module)]=original.create(module.weight)
    old=original.run(handles[id(module)],x,module.out_features)
    if module.bias is not None:old+=module.bias.float()
    old=old.to(x.dtype)
    engine.fusion=False
    try:legacy=execute(module,x,**kwargs)
    finally:engine.fusion=True
    fused=execute(module,x,**kwargs)
    # Defer tensor comparisons until inference ends, so comparison kernels do
    # not change the CPU execution state on the hot path under investigation.
    captured.append((names[id(module)],list(x.shape),old,legacy,fused,initial,backend()))
    return fused
engine.execute=shadow
try:
    case=next(c for c in json.loads(Path('/references/request-cases.json').read_text()) if c['id']=='english')
    response=model.system_one(**{k:case[k] for k in ('state','questions','images','videos') if k in case})
    records=[]
    for name,shape,old,legacy,fused,before,after in captured:
        records.append({'name':name,'input_shape':shape,'old_vs_new_legacy_equal':torch.equal(old,legacy),
                        'new_legacy_vs_fused_equal':torch.equal(legacy,fused),
                        'max_old_vs_fused_abs':(old.float()-fused.float()).abs().max().item(),
                        'old_vs_fused_changed_elements':int((old!=fused).sum()),'backend_before':before,'backend_after':after})
    result={'passed':all(r['old_vs_new_legacy_equal'] and r['new_legacy_vs_fused_equal'] for r in records),
            'scope':'Shadow one real English three-question inference; 78 large projections execute original bridge, new legacy bridge, and fused bridge on the same input. Comparisons deferred until end. No parallel scheduling.',
            'records':records,'response':response,'backend_final':backend()}
    Path('/work/logs/shadow-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('SHADOW_RESULT',result['passed'],'linear_calls',len(records),'backend',backend(),flush=True)
finally:
    restore_adapter();engine.close();original.close()
    for callback in reversed(restore_native):callback()
