"""Low-overhead wall/CPU attribution; no Torch profiler trace collection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import resource
import sys
import time
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from cpu_attention_optimized import install_conv, install_scan, install_sdpa

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--count', type=int, default=30)
a = p.parse_args()
sys.path[:0] = ['/base', a.model]
from cpu_optimized_server import optimize
from rknn_linear import Engine, install, uninstall
from d3_runtime import D3
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

torch.set_num_threads(4)
model = D3.from_pretrained(a.model, device='cpu', verify='full')
optimize(model)
engine = Engine()
install(model, engine)
restore = [install_conv(), install_scan(), install_sdpa()]
body = json.loads(Path('/base/request.json').read_text())
def invoke():
    return model.system_one(state=body['state'], questions=body['questions'])

stats, operators, slow = {}, {}, []
class ScanOps(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        t = time.perf_counter()
        out = func(*args, **(kwargs or {}))
        elapsed = time.perf_counter()-t
        name = str(func)
        operators[name] = operators.get(name, 0) + elapsed
        if elapsed > .015:
            slow.append({'operator': name, 'seconds': elapsed,
                         'shapes': [list(x.shape) for x in args if isinstance(x, torch.Tensor)]})
        return out

wrapped = []
def wrap(owner, name, scan=False):
    original = getattr(owner, name)
    def call(*args, **kwargs):
        t = time.perf_counter()
        if scan:
            with ScanOps():
                out = original(*args, **kwargs)
        else:
            out = original(*args, **kwargs)
        stats[name] = stats.get(name, 0) + time.perf_counter()-t
        return out
    setattr(owner, name, call)
    wrapped.append((owner, name, original))

records = []
try:
    invoke()
    wrap(qwen, 'torch_chunk_gated_delta_rule', True)
    wrap(qwen, 'causal_conv1d_fn')
    wrap(torch.nn.functional, 'scaled_dot_product_attention')
    wrap(engine, 'run')
    for i in range(a.count):
        stats, operators, slow = {}, {}, []
        before = resource.getrusage(resource.RUSAGE_SELF)
        t = time.perf_counter()
        response = invoke()
        elapsed = time.perf_counter()-t
        after = resource.getrusage(resource.RUSAGE_SELF)
        record = dict(sample=i, seconds=elapsed, components=stats, operators=operators,
                      slow_operators=slow, user_seconds=after.ru_utime-before.ru_utime,
                      system_seconds=after.ru_stime-before.ru_stime,
                      voluntary_switches=after.ru_nvcsw-before.ru_nvcsw,
                      involuntary_switches=after.ru_nivcsw-before.ru_nivcsw,
                      minor_faults=after.ru_minflt-before.ru_minflt,
                      choice=response['answers']['kind']['choice'])
        records.append(record)
        print(json.dumps(record), flush=True)
    Path('/work/logs/tail-ops.json').write_text(json.dumps(records, indent=2)+'\n')
finally:
    for owner, name, original in reversed(wrapped):
        setattr(owner, name, original)
    for callback in reversed(restore):
        callback()
    uninstall(model)
    engine.close()
