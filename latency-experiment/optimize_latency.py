"""Compare narrowly scoped CPU kernel substitutions in an isolated process."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import runpy
import torch
from cpu_attention_optimized import install_conv, install_scan, install_sdpa

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
args = parser.parse_args()
sys.path[:0] = ['/base', args.model]
runpy.run_path('/work/test_causal_conv.py')
runpy.run_path('/work/test_scan.py')
from cpu_optimized_server import optimize
from rknn_linear import Engine, install, uninstall
from d3_runtime import D3, PromptEncoder

torch.set_num_threads(4)
model = D3.from_pretrained(args.model, device='cpu', verify='full')
assert model.attention_mode == 'noncausal_full_attention' and isinstance(model.encoder, PromptEncoder)
optimize(model)
engine = Engine()
install(model, engine)
request = json.loads(Path('/base/request.json').read_text())
results = {'variants': [], 'model_unchanged': True, 'attention_mode': model.attention_mode}
destination = Path('/work/logs/optimization-result.json')
def invoke():
    return model.system_one(state=request['state'], questions=request['questions'])
def numbers(x, path=''):
    result = {}
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in ('confidence', 'legend'):
                result.update(numbers(v, path+'/'+k))
    elif isinstance(x, (int, float)):
        result[path] = x
    return result
restore = []
try:
    reference = invoke()
    def measure(label):
        invoke()
        before = engine.seconds
        times = []
        for _ in range(3):
            start = time.perf_counter()
            response = invoke()
            times.append(time.perf_counter()-start)
        a, b = numbers(response['answers']), numbers(reference['answers'])
        delta = max(abs(a[k]-b[k]) for k in a)
        choices = all(v.get('choice') == reference['answers'][k].get('choice') for k,v in response['answers'].items())
        item = {'variant': label, 'threads': torch.get_num_threads(), 'seconds': times,
                'median': statistics.median(times), 'max_delta_vs_original_rknn': delta,
                'choices_unchanged': choices, 'adapter_per_call': (engine.seconds-before)/3,
                'passed': delta <= 0.01 and choices, 'response': response}
        results['variants'].append(item)
        destination.write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k:v for k,v in item.items() if k!='response'}), flush=True)
    measure('original')
    restore.append(install_conv())
    measure('vectorized_conv')
    restore.append(install_scan())
    measure('vectorized_conv+skip_zero_unused_scan_state')
    restore.append(install_sdpa())
    measure('conv+scan+fp32_sdpa')
    torch.set_num_threads(2)
    measure('conv+scan+fp32_sdpa')
    torch.set_num_threads(1)
    measure('conv+scan+fp32_sdpa')
    results['completed'] = True
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
finally:
    for callback in reversed(restore):
        callback()
    uninstall(model)
    engine.close()
