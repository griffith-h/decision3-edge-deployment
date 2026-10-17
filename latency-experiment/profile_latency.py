"""Independent profiling: unchanged checkpoint, attention policy, original service."""
import argparse
import collections
import inspect
import json
from pathlib import Path
import statistics
import sys
import time

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--operator-profile', action='store_true', help='High-overhead optional Kineto trace')
args = parser.parse_args()
sys.path[:0] = ['/base', args.model]
from cpu_optimized_server import optimize
from rknn_linear import Engine, install, uninstall
from d3_runtime import D3, PromptEncoder
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

out = Path('/work/logs')
out.mkdir(exist_ok=True)
torch.set_num_threads(4)
start = time.perf_counter()
model = D3.from_pretrained(args.model, device='cpu', verify='full')
assert model.attention_mode == 'noncausal_full_attention'
assert isinstance(model.encoder, PromptEncoder)
optimize(model)
engine = Engine()
install(model, engine)
body = json.loads(Path('/base/request.json').read_text())


def invoke():
    return model.system_one(state=body['state'], questions=body['questions'])


def numeric(value, path=''):
    found = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in ('confidence', 'legend'):
                found.update(numeric(child, path + '/' + key))
    elif isinstance(value, (int, float)):
        found[path] = value
    return found


reference = None
results = {'checkpoint': args.model, 'attention_mode': model.attention_mode,
           'torch': torch.__version__, 'model_structure_changed': False,
           'measurements': [], 'profile': {}}


def measure(name, repeats=None):
    invoke()
    before = engine.seconds
    timings = []
    for _ in range(repeats or args.repeats):
        t = time.perf_counter()
        response = invoke()
        timings.append(time.perf_counter() - t)
    actual, expected = numeric(response['answers']), numeric(reference['answers'])
    assert actual.keys() == expected.keys()
    delta = max(abs(actual[k] - expected[k]) for k in actual)
    choices = all(v.get('choice') == reference['answers'][k].get('choice') for k, v in response['answers'].items())
    record = {'variant': name, 'threads': torch.get_num_threads(), 'seconds': timings,
              'median': statistics.median(timings), 'mean': statistics.mean(timings),
              'adapter_per_request_seconds': (engine.seconds - before) / len(timings),
              'max_delta': delta, 'choices_unchanged': choices, 'response': response}
    results['measurements'].append(record)
    (out / 'profile-result.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'response'}), flush=True)
    return record


try:
    reference = invoke()
    print('MODEL_READY', time.perf_counter() - start, flush=True)
    for threads in (4, 2, 1):
        torch.set_num_threads(threads)
        measure('original')

    torch.set_num_threads(4)
    modules = {}
    handles = []
    starts = collections.defaultdict(list)
    totals = collections.defaultdict(lambda: {'seconds': 0.0, 'calls': 0})
    for name, module in model.backbone.named_modules():
        if ('language_model' in name and
            (name.endswith(('.linear_attn', '.self_attn', '.mlp', '.input_layernorm', '.post_attention_layernorm', '.norm'))
             or 'RMSNorm' in type(module).__name__)):
            modules[name] = module
            def before(module, inputs, key=name):
                starts[key].append(time.perf_counter())
            def after(module, inputs, output, key=name):
                totals[key]['seconds'] += time.perf_counter() - starts[key].pop()
                totals[key]['calls'] += 1
            handles.extend([module.register_forward_pre_hook(before), module.register_forward_hook(after)])
    functions = []
    for owner, name in [(qwen, 'torch_chunk_gated_delta_rule'), (qwen, 'causal_conv1d_fn'),
                        (qwen, 'apply_rotary_pos_emb'), (torch.nn.functional, 'scaled_dot_product_attention')]:
        if not hasattr(owner, name):
            continue
        original = getattr(owner, name)
        def wrapped(*pos, _original=original, _name=name, **kwargs):
            t = time.perf_counter()
            try:
                return _original(*pos, **kwargs)
            finally:
                totals['function:' + _name]['seconds'] += time.perf_counter() - t
                totals['function:' + _name]['calls'] += 1
        functions.append((owner, name, original))
        setattr(owner, name, wrapped)
    t = time.perf_counter()
    invoke()
    results['profile']['instrumented_total'] = time.perf_counter() - t
    results['profile']['nested_module_and_function_wall_seconds'] = dict(totals)
    for h in handles:
        h.remove()
    for owner, name, original in functions:
        setattr(owner, name, original)
    print('MODULE_PROFILE', json.dumps(dict(totals)), flush=True)
    (out / 'profile-result.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    if args.operator_profile:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False) as prof:
            invoke()
        operators = []
        for op in prof.key_averages():
            operators.append({'op': op.key, 'self_cpu_seconds': op.self_cpu_time_total / 1e6,
                              'cpu_total_seconds': op.cpu_time_total / 1e6, 'calls': op.count})
        results['profile']['operators'] = sorted(operators, key=lambda r: r['self_cpu_seconds'], reverse=True)
        (out / 'operators.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=45))
        print('OPERATOR_PROFILE', json.dumps(results['profile']['operators'][:20]), flush=True)

    original_sdpa = torch.nn.functional.scaled_dot_product_attention
    def fp32_sdpa(query, key, value, *pos, **kwargs):
        dtype = query.dtype
        if query.device.type == 'cpu' and dtype == torch.bfloat16:
            mask = kwargs.get('attn_mask')
            if mask is not None and mask.dtype != torch.bool:
                kwargs['attn_mask'] = mask.float()
            return original_sdpa(query.float(), key.float(), value.float(), *pos, **kwargs).to(dtype)
        return original_sdpa(query, key, value, *pos, **kwargs)
    torch.nn.functional.scaled_dot_product_attention = fp32_sdpa
    try:
        for threads in (4, 2, 1):
            torch.set_num_threads(threads)
            measure('fp32_sdpa')
    finally:
        torch.nn.functional.scaled_dot_product_attention = original_sdpa
    results['completed'] = True
    (out / 'profile-result.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
finally:
    uninstall(model)
    engine.close()
