"""Paired correctness checks, lightweight timing, then a temporary HTTP probe."""
import argparse
from concurrent.futures import Future
import copy
import json
from pathlib import Path
import queue
import runpy
import statistics
import sys
import threading
import time
import torch
from cpu_attention_optimized import install_conv, install_scan, install_sdpa

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--port', type=int, default=8901)
parser.add_argument('--chunk-size', type=int)
parser.add_argument('--native-norm', action='store_true')
parser.add_argument('--native-conv', action='store_true')
parser.add_argument('--output', default='validation-result.json')
parser.add_argument('--no-serve', action='store_true')
args = parser.parse_args()
sys.path[:0] = ['/base', args.model]
for filename in ('test_causal_conv.py', 'test_scan.py', 'test_sdpa.py'):
    runpy.run_path('/work/' + filename)
if args.chunk_size:
    runpy.run_path('/work/test_chunk_sizes.py')
if args.native_norm:
    runpy.run_path('/work/test_native_norm.py')
if args.native_conv:
    runpy.run_path('/work/test_native_conv.py')
from cpu_optimized_server import optimize
from rknn_linear import Engine, install, uninstall
from d3_runtime import D3, PromptEncoder
from d3_server import serve
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

torch.set_num_threads(4)
model = D3.from_pretrained(args.model, device='cpu', verify='full')
model.model_name = 'vllm-sr/d3-edge'
assert model.attention_mode == 'noncausal_full_attention' and isinstance(model.encoder, PromptEncoder)
optimize(model)
engine = Engine()
install(model, engine)
cases = json.loads(Path('/base/request-cases.json').read_text())
image = json.loads(Path('/base/golden-reference.json').read_text())['image']
image['state'] = 'Describe the attached image.'
cases.append({'id': 'image', **image})
single = next(c for c in cases if c['id'] == 'single')
for name, text in [('novel-math', '请计算 37 乘以 19，并检查结果。'),
                   ('novel-chat', '今天下班后想和朋友散步聊天，有什么轻松的话题？')]:
    item = copy.deepcopy(single)
    item['id'], item['state'] = name, text
    cases.append(item)
reference_cpu = json.loads(Path('/base/logs/benchmark-all.json').read_text())['cases']
results = {'model_unchanged': True, 'attention_mode': model.attention_mode,
           'chunk_size_override': args.chunk_size, 'native_norm': args.native_norm,
           'native_conv': args.native_conv,
           'model_layers': len(model.encoder.stack), 'cases': {}, 'profiles': {}}
destination = Path('/work/logs', args.output)
def save():
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
def invoke(case):
    return model.system_one(**{k:case[k] for k in ('state', 'questions', 'images', 'videos') if k in case})
def numeric(x, path=''):
    found = {}
    if isinstance(x, dict):
        for key, value in x.items():
            if key not in ('confidence', 'legend'):
                found.update(numeric(value, path+'/'+key))
    elif isinstance(x, (int, float)):
        found[path] = x
    return found
def difference(a, b):
    an, bn = numeric(a['answers']), numeric(b['answers'])
    assert an.keys() == bn.keys()
    delta = max(abs(an[k]-bn[k]) for k in an)
    choices = all(v.get('choice') == b['answers'][k].get('choice') for k,v in a['answers'].items())
    return delta, choices
def profile(label):
    stats = {}
    functions = []
    hooks = []
    def wrap(owner, name):
        original = getattr(owner, name)
        def forward(*pos, **kwargs):
            t = time.perf_counter()
            try:
                return original(*pos, **kwargs)
            finally:
                stats[name] = stats.get(name, 0) + time.perf_counter()-t
        functions.append((owner, name, original))
        setattr(owner, name, forward)
    for owner, name in ((qwen, 'causal_conv1d_fn'), (qwen, 'torch_chunk_gated_delta_rule'),
                        (torch.nn.functional, 'scaled_dot_product_attention'), (engine, 'run'),
                        (model, 'prepare'), (model, 'respond')):
        wrap(owner, name)
    starts = {}
    for name, module in model.backbone.named_modules():
        if 'RMSNorm' in type(module).__name__:
            def before(module, inputs, key=name):
                starts[key] = time.perf_counter()
            def after(module, inputs, output, key=name):
                stats['rmsnorm'] = stats.get('rmsnorm', 0) + time.perf_counter()-starts[key]
            hooks += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    try:
        t = time.perf_counter()
        invoke(single)
        stats['total'] = time.perf_counter()-t
    finally:
        for owner, name, original in reversed(functions):
            setattr(owner, name, original)
        for hook in hooks:
            hook.remove()
    known = ['causal_conv1d_fn', 'torch_chunk_gated_delta_rule', 'scaled_dot_product_attention',
             'run', 'prepare', 'respond', 'rmsnorm']
    stats['other'] = stats['total'] - sum(stats.get(k, 0) for k in known)
    results['profiles'][label] = stats
    save()
    print('PROFILE', label, json.dumps(stats), flush=True)

restore = []
native_stats = {}
server = None
http_thread = None
try:
    invoke(single)
    for case in cases:
        t = time.perf_counter()
        response = invoke(case)
        results['cases'][case['id']] = {'original_rknn': {'seconds': time.perf_counter()-t, 'response': response}}
        save()
    profile('original_rknn')
    restore = [install_conv(), install_scan(args.chunk_size), install_sdpa()]
    if args.native_norm:
        from native_norm import install_norm
        restore.append(install_norm(model.backbone, native_stats))
    if args.native_conv:
        from native_norm import install_native_conv
        restore.append(install_native_conv(native_stats))
    for case in cases:
        invoke(case)
        timings = []
        for _ in range(3):
            t = time.perf_counter()
            response = invoke(case)
            timings.append(time.perf_counter()-t)
        old = results['cases'][case['id']]['original_rknn']['response']
        delta, choices = difference(response, old)
        record = {'seconds': timings, 'median': statistics.median(timings),
                  'max_delta_vs_original_rknn': delta, 'choices_unchanged': choices, 'response': response}
        assert delta <= 0.01 and choices, (case['id'], delta, choices)
        if case['id'] in reference_cpu:
            delta_cpu, choices_cpu = difference(response, reference_cpu[case['id']]['cpu']['response'])
            record.update(max_delta_vs_historical_optimized_cpu=delta_cpu, choices_unchanged_vs_cpu=choices_cpu)
            assert delta_cpu <= 0.01 and choices_cpu, (case['id'], delta_cpu, choices_cpu)
        results['cases'][case['id']]['candidate'] = record
        save()
        print('VALIDATED', case['id'], json.dumps({k:v for k,v in record.items() if k!='response'}), flush=True)
    profile('candidate')
    if args.native_norm:
        assert native_stats.get('norm_calls', 0) > 0
    if args.native_conv:
        assert native_stats.get('conv_calls', 0) > 0
    results['native_stats'] = native_stats
    results['passed'] = True
    save()
    if args.no_serve:
        sys.exit(0)
    server = serve(model, '0.0.0.0', args.port)
    answer = server.decider.answer
    pending = queue.Queue()
    def queued(body):
        future = Future()
        pending.put((body, future))
        return future.result()
    server.decider.answer = queued
    http_thread = threading.Thread(target=server.serve_forever, daemon=True)
    http_thread.start()
    print('TEMPORARY_HTTP_READY', args.port, flush=True)
    while True:
        body, future = pending.get()
        try:
            future.set_result(answer(body))
        except Exception as exc:
            future.set_exception(exc)
        finally:
            pending.task_done()
finally:
    if server:
        if http_thread:
            server.shutdown()
        server.server_close()
    for callback in reversed(restore):
        callback()
    uninstall(model)
    engine.close()
