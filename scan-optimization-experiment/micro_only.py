"""Replay locally captured real scan tensors; no model or RKNN allocation."""
import hashlib,json,statistics,sys,time
from pathlib import Path
import torch
sys.path[:0]=['/release','/base','/work']
from cpu_attention_optimized import chunk_delta_prefill
from scan import make_scan
from test_scan import run
torch.set_num_threads(3)
run()
def baseline(*args,**kw):
    kw.pop('chunk_size',None)
    return chunk_delta_prefill(*args,**kw,chunk_size=128)
captured=torch.load('/work/logs/captured-scans.pt',weights_only=True)
variants={'baseline':baseline,'compact':make_scan(baseline,'compact')}
for tile in (16,32,64,128):
    for fused in (0,1,2,3,4,5):variants[f'native_t{tile}_f{fused}']=make_scan(baseline,tile=tile,fused=fused)
result={'sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').iterdir() if p.suffix in ('.py','.c','.so')},'variants':{}}
with torch.inference_mode():
    for name,fn in variants.items():
        records=[]
        for args,kw,ref in captured:
            times=[]
            for _ in range(11):
                t=time.perf_counter();actual=fn(*args,**kw)[0];times.append(time.perf_counter()-t)
            d=(actual.float()-ref.float()).abs()
            records.append({'median':statistics.median(times),'seconds':times,'max_abs':d.max().item(),
                            'relative_l2':(torch.linalg.vector_norm(d)/torch.linalg.vector_norm(ref.float())).item(),
                            'bitwise_equal':torch.equal(actual,ref),'finite':bool(torch.isfinite(actual).all())})
        result['variants'][name]={'layers':records,'sum_medians':sum(r['median'] for r in records),'counts':getattr(fn,'counts',None)}
        if name!='baseline':assert fn.counts=={'optimized':66,'fallback':0},(name,fn.counts)
        print('REPLAY',name,result['variants'][name]['sum_medians'],max(r['max_abs'] for r in records),getattr(fn,'counts',None),flush=True)
Path('/work/logs/replay-results.json').write_text(json.dumps(result,indent=2)+'\n')
