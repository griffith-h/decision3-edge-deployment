"""Compare chunk sizes and optional native norm on the exact same weights."""
import argparse
import json
from pathlib import Path
import resource
import runpy
import statistics
import sys
import time
import torch
from cpu_attention_optimized import install_conv, install_scan, install_sdpa
from native_norm import install_norm, install_native_conv

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--conv-study', action='store_true')
a = p.parse_args()
sys.path[:0] = ['/base', a.model]
from cpu_optimized_server import optimize
from rknn_linear import Engine, install, uninstall
from d3_runtime import D3

runpy.run_path('/work/test_native_norm.py')
if a.conv_study:
    runpy.run_path('/work/test_native_conv.py')
torch.set_num_threads(4)
model = D3.from_pretrained(a.model, device='cpu', verify='full')
optimize(model)
engine = Engine()
install(model, engine)
restores = [install_conv(), install_sdpa()]
body = json.loads(Path('/base/request.json').read_text())
def invoke():
    return model.system_one(state=body['state'], questions=body['questions'])

results = {}
reference = None
try:
    variants = ((128, True, False), (128, True, True)) if a.conv_study else ((64, False, False), (128, False, False), (256, False, False), (128, True, False))
    for chunk, native, native_conv in variants:
        scan_restore = install_scan(chunk)
        norm_restore = install_norm(model.backbone) if native else lambda: None
        conv_restore = install_native_conv() if native_conv else lambda: None
        name = f'chunk{chunk}-norm{native}' + ('-convTrue' if native_conv else '')
        records = []
        try:
            invoke()
            for i in range(12):
                before = resource.getrusage(resource.RUSAGE_SELF)
                t = time.perf_counter()
                output = invoke()
                elapsed = time.perf_counter()-t
                after = resource.getrusage(resource.RUSAGE_SELF)
                if reference is None:
                    reference = output
                probs = output['answers']['kind']['probabilities']
                expected = reference['answers']['kind']['probabilities']
                delta = max(abs(probs[k]-expected[k]) for k in probs)
                assert delta <= .01 and output['answers']['kind']['choice'] == reference['answers']['kind']['choice']
                records.append(dict(seconds=elapsed, max_delta=delta,
                                    minor_faults=after.ru_minflt-before.ru_minflt,
                                    voluntary_switches=after.ru_nvcsw-before.ru_nvcsw,
                                    involuntary_switches=after.ru_nivcsw-before.ru_nivcsw))
                print(name, i, json.dumps(records[-1]), flush=True)
            times = [r['seconds'] for r in records]
            results[name] = dict(records=records, median=statistics.median(times), maximum=max(times), response=output)
            Path('/work/logs', 'conv-tuning.json' if a.conv_study else 'cpu-tuning.json').write_text(json.dumps(results, indent=2)+'\n')
        finally:
            conv_restore()
            norm_restore()
            scan_restore()
finally:
    for restore in reversed(restores):
        restore()
    uninstall(model)
    engine.close()
