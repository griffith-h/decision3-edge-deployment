"""Focused throughput screening; synthetic arrays, not an end-to-end result."""
import sys,json,time,statistics
from pathlib import Path
sys.path[:0]=['/release','/base','/work']
import torch
torch.set_num_threads(3)
from sleef_setup import prepare
prepare()
from cpu_math import activation
from test_math import run
unit=run();torch.manual_seed(175)
x=torch.randn(115,3584,dtype=torch.bfloat16)*8;up=torch.randn_like(x)
result={'scope':'Synthetic throughput screening, 115x3584 BF16; not real-input or model latency. Open-source installed Torch public SLEEF export, no vendor inspection.','units':unit,'timings':{}}
choices={'torch_silu':lambda:torch.nn.functional.silu(x),'sleef_silu':lambda:activation(x),
         'torch_silu_mul':lambda:torch.nn.functional.silu(x)*up,'sleef_silu_mul':lambda:activation(x,up,1)}
with torch.inference_mode():
    for name,fn in choices.items():
        times=[]
        for _ in range(31):
            t=time.perf_counter();fn();times.append(time.perf_counter()-t)
        result['timings'][name]={'median':statistics.median(times[1:]),'seconds':times}
        print('SLEEF',name,result['timings'][name]['median'],flush=True)
Path('/work/logs/sleef-screen.json').write_text(json.dumps(result,indent=2)+'\n')
