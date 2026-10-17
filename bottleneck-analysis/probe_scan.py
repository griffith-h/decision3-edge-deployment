"""Operator wall clocks on real six-layer scan inputs; no arithmetic replacement."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
sys.path[:0]=['/release','/base',a.model]
from d3_runtime import D3
from cpu_optimized_server import optimize
from rknn_linear import Engine,install,uninstall
from cpu_attention_optimized import install_conv,install_scan,install_sdpa
from native_norm import install_norm,install_native_conv
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
torch.set_num_threads(3)
model=D3.from_pretrained(a.model,device='cpu',verify='full');optimize(model)
engine=Engine();install(model,engine)
stats={};restore=[install_conv(),install_scan(128),install_sdpa(),install_norm(model.backbone,stats),install_native_conv(stats)]
original=qwen.torch_chunk_gated_delta_rule
captured=[]
def capture(*args,**kwargs):
    output=original(*args,**kwargs)
    captured.append((args,kwargs,output[0]))
    return output
qwen.torch_chunk_gated_delta_rule=capture
operators=defaultdict(lambda:{'seconds':0.,'calls':0,'shapes':[]})
class Ops(TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        t=time.perf_counter();out=func(*args,**(kwargs or {}));elapsed=time.perf_counter()-t
        row=operators[str(func)];row['seconds']+=elapsed;row['calls']+=1
        shapes=[list(v.shape) for v in args if isinstance(v,torch.Tensor)]
        if shapes not in row['shapes'] and len(row['shapes'])<3:row['shapes'].append(shapes)
        return out
try:
    body=json.loads(Path('/base/request.json').read_text());response=model.system_one(state=body['state'],questions=body['questions'])
    assert len(captured)==6
    qwen.torch_chunk_gated_delta_rule=original
    result={'scope':'Microprofile of six actual GatedDeltaNet inputs from one 115-token request; operator instrumentation adds overhead. Not an end-to-end latency benchmark.','cases':[],'response':response}
    for i,(args,kwargs,reference) in enumerate(captured):
        values=[]
        for _ in range(5):
            t=time.perf_counter()
            with torch.inference_mode():actual=original(*args,**kwargs)[0]
            values.append(time.perf_counter()-t)
            assert torch.equal(actual,reference)
        operators.clear();t=time.perf_counter()
        with torch.inference_mode(),Ops():actual=original(*args,**kwargs)[0]
        elapsed=time.perf_counter()-t
        assert torch.equal(actual,reference)
        record={'layer_scan_index':i,'unprofiled_seconds':values,'median':statistics.median(values),
                'profiled_seconds':elapsed,'operators':dict(operators),'outputs_bitwise_equal':True,
                'argument_shapes':[list(x.shape) for x in args if isinstance(x,torch.Tensor)]}
        result['cases'].append(record);print('SCAN',i,record['median'],elapsed,flush=True)
    result['passed']=True
    Path('/work/logs/scan-operators.json').write_text(json.dumps(result,indent=2)+'\n')
finally:
    qwen.torch_chunk_gated_delta_rule=original
    for callback in reversed(restore):callback()
    uninstall(model);engine.close()
