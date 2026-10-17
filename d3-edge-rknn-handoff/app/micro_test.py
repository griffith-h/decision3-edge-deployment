import json
import time
from pathlib import Path
import torch
from rknn_linear import Engine

torch.set_num_threads(4)
torch.manual_seed(7)
engine = Engine()
results = []
try:
    for k,n in [(1024,3072),(3072,1024),(1024,2048),(128,64)]:
        # BF16 model values are exactly representable in FP16 within its exponent range.
        w = (torch.randn(n,k) * 0.02).bfloat16().float()
        handle = engine.create(w)
        for m in [115,129,97,381]:
            x = torch.randn(m,k).bfloat16()
            ref = torch.nn.functional.linear(x.float(),w)
            y = engine.run(handle,x,n)
            delta = y-ref
            assert torch.isfinite(y).all()
            relative = delta.norm().item()/ref.norm().item()
            assert relative < 0.005, (m,k,n,relative)
            cpu=[]; npu=[]
            for _ in range(5):
                t=time.perf_counter(); torch.nn.functional.linear(x.float(),w); cpu.append(time.perf_counter()-t)
                t=time.perf_counter(); engine.run(handle,x,n); npu.append(time.perf_counter()-t)
            r={'M':m,'K':k,'N':n,'max_absolute_delta':delta.abs().max().item(),'relative_l2_delta':relative,
               'cpu_seconds_median':sorted(cpu)[2],'npu_seconds_median':sorted(npu)[2]}
            r['speedup']=r['cpu_seconds_median']/r['npu_seconds_median']
            results.append(r); print(json.dumps(r),flush=True)
    Path('/work/logs/micro-test.json').write_text(json.dumps(results,indent=2)+'\n')
finally:
    engine.close()
