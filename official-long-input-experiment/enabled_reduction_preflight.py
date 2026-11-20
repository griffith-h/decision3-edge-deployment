"""CPU-only reduction qualification against the unchanged enabled backend.

Synthetic intermediates exercise dimensions; no model, SDK context, labels or
benchmark answers are loaded. This never qualifies full-model accuracy.
"""
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as f
from sparse_cpu_backend import DiagonalRepair

def run(emit,out):
    torch.set_num_threads(3)
    if not torch.backends.mkldnn.enabled:raise AssertionError('Original enabled CPU backend required')
    (out/'cpu-preflight-identity.json').write_text(json.dumps({'torch':str(torch.__version__),
        'torch_git':torch.version.git_version,'threads':torch.get_num_threads(),
        'mkldnn':torch.backends.mkldnn.enabled,'float32_precision':torch.get_float32_matmul_precision(),
        'lab_sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in Path(__file__).parent.iterdir() if p.is_file()}})+'\n')
    generator=torch.Generator().manual_seed(65)
    records=[]
    emit({'event':'ready','case':'enabled-cpu-reduction-preflight','model_loaded':False})
    emit({'event':'request_begin','phase':'synthetic_enabled_cpu_reductions'})
    with torch.inference_mode():
        for k,n in ((64,256),(1024,2048),(2048,1024),(3584,1024),(6144,512),(10240,128)):
            x=torch.randn(129,k,generator=generator,dtype=torch.bfloat16)
            weight=torch.randn(n,k,generator=generator,dtype=torch.bfloat16)
            module=SimpleNamespace(weight=weight)
            expected=f.linear(x.float(),weight.float()).bfloat16()
            mask=torch.rand(129,n,generator=generator)<.15;pairs=mask.nonzero()
            variants=[]
            for block in (352,1024,16384):
                actual=torch.full_like(expected,77.);start=time.perf_counter()
                DiagonalRepair(block).run(module,x,pairs,actual)
                variants.append({'qblock':block,'seconds':time.perf_counter()-start,
                    'selected':len(pairs),'bf16_mismatches':int((actual[mask]!=expected[mask]).sum()),
                    'unselected_untouched':bool((actual[~mask]==77).all())})
            records.append({'shape':[129,k,n],'variants':variants})
            (out/'enabled-reduction-preflight.json').write_text(json.dumps({'completed':False,
                'scope':'CPU synthetic reduction check only; no model or benchmark accuracy','records':records})+'\n')
    if not torch.backends.mkldnn.enabled:raise AssertionError('CPU backend changed during preflight')
    data={'completed':True,'model_loaded':False,'mkldnn':True,
          'scope':'CPU synthetic reduction check only; no model or benchmark accuracy','records':records,
          'qualified_blocks':[block for block in (352,1024,16384)
              if all(v['bf16_mismatches']==0 and v['unselected_untouched']
                     for r in records for v in r['variants'] if v['qblock']==block)]}
    (out/'enabled-reduction-preflight.json').write_text(json.dumps(data)+'\n')
    emit({'event':'complete','model_loaded':False,'dimensions':len(records)})
