"""Investigate a slow instrumented sample without changing online services."""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import torch
from cpu_attention_optimized import install_conv, install_scan, install_sdpa

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--count', type=int, default=40)
parser.add_argument('--chunk-size', type=int)
parser.add_argument('--native-norm', action='store_true')
parser.add_argument('--native-conv', action='store_true')
parser.add_argument('--output', default='tail-investigation.json')
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--check-cases', action='store_true')
parser.add_argument('--case-reference', default='/work/logs/validation-final.json')
parser.add_argument('--serve-after', action='store_true')
args = parser.parse_args()
sys.path[:0] = ['/base', args.model]
from cpu_optimized_server import optimize
from rknn_linear import Engine, install, uninstall
from d3_runtime import D3
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
torch.set_num_threads(args.threads)
backend = {'parallel_info': torch.__config__.parallel_info()}
for line in Path('/proc/self/maps').read_text().splitlines():
    if 'libopenblas' in line and '/' in line:
        lib = ctypes.CDLL(line.split()[-1])
        for name in ('openblas_get_num_threads', 'openblas_get_parallel', 'openblas_get_config'):
            if hasattr(lib, name):
                fn = getattr(lib, name)
                fn.restype = ctypes.c_char_p if name.endswith('config') else ctypes.c_int
                value = fn()
                backend[name] = value.decode() if isinstance(value, bytes) else value
        break
print('BACKEND', json.dumps(backend), flush=True)
model = D3.from_pretrained(args.model, device='cpu', verify='full')
optimize(model)
engine = Engine()
install(model, engine)
restore = [install_conv(), install_scan(args.chunk_size), install_sdpa()]
native_stats = {}
if args.native_norm:
    from native_norm import install_norm
    restore.append(install_norm(model.backbone, native_stats))
if args.native_conv:
    from native_norm import install_native_conv
    restore.append(install_native_conv(native_stats))
body = json.loads(Path('/base/request.json').read_text())
events = []
gc_starts = {}
sample = -1
def on_gc(phase, info):
    generation = info['generation']
    if phase == 'start':
        gc_starts[generation] = time.perf_counter()
    else:
        events.append({'sample': sample, 'generation': generation,
                       'seconds': time.perf_counter()-gc_starts[generation],
                       'collected': info['collected']})
records = []
try:
    def invoke():
        return model.system_one(state=body['state'], questions=body['questions'])
    invoke()
    if args.native_norm:
        assert native_stats.get('norm_calls', 0) > 0, 'Native norm was not actually dispatched'
    if args.native_conv:
        assert native_stats.get('conv_calls', 0) > 0, 'Native conv was not actually dispatched'
    gc.callbacks.append(on_gc)
    for sample in range(args.count):
        frequency_path = Path('/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq')
        frequency_before = int(frequency_path.read_text()) if frequency_path.exists() else None
        t = time.perf_counter()
        adapter = engine.seconds
        before = resource.getrusage(resource.RUSAGE_SELF)
        response = invoke()
        elapsed = time.perf_counter()-t
        after = resource.getrusage(resource.RUSAGE_SELF)
        record = {'sample': sample, 'seconds': elapsed, 'adapter_seconds': engine.seconds-adapter,
                  'frequency_before_khz': frequency_before,
                  'frequency_after_khz': int(frequency_path.read_text()) if frequency_path.exists() else None,
                  'minor_faults': after.ru_minflt-before.ru_minflt, 'major_faults': after.ru_majflt-before.ru_majflt,
                  'user_seconds': after.ru_utime-before.ru_utime, 'system_seconds': after.ru_stime-before.ru_stime,
                  'voluntary_switches': after.ru_nvcsw-before.ru_nvcsw,
                  'involuntary_switches': after.ru_nivcsw-before.ru_nivcsw,
                  'gc_seconds': sum(e['seconds'] for e in events if e['sample']==sample),
                  'choice': response['answers']['kind']['choice']}
        records.append(record)
        print(json.dumps(record), flush=True)
    case_checks = None
    if args.check_cases:
        from check_cases import validate_cases
        case_checks = validate_cases(model, args.case_reference)
    Path('/work/logs', args.output).write_text(json.dumps({'records': records, 'gc_events': events,
        'gc_disabled': False, 'chunk_size_override': args.chunk_size,
        'native_norm': args.native_norm, 'native_conv': args.native_conv,
        'native_stats': native_stats, 'engine_calls': engine.calls, 'cpu_fallbacks': engine.cpu_fallbacks,
        'backend': backend,
        'case_checks': case_checks,
        'source_sha256': {name: hashlib.sha256(Path('/work', name).read_bytes()).hexdigest()
                          for name in ('cpu_attention_optimized.py', 'native_norm.py', 'native_norm.c')},
        'settings': {'torch_threads': torch.get_num_threads(), 'torch_interop_threads': torch.get_num_interop_threads(),
                     'affinity': sorted(os.sched_getaffinity(0)),
                     'OMP_THREAD_LIMIT': os.environ.get('OMP_THREAD_LIMIT'),
                     'OMP_WAIT_POLICY': os.environ.get('OMP_WAIT_POLICY'),
                     'GOMP_SPINCOUNT': os.environ.get('GOMP_SPINCOUNT')}}, indent=2)+'\n')
    if args.serve_after:
        from temporary_http import serve_on_main_thread
        model.model_name = 'vllm-sr/d3-edge'
        serve_on_main_thread(model, 8901)
finally:
    if on_gc in gc.callbacks:
        gc.callbacks.remove(on_gc)
    for callback in reversed(restore):
        callback()
    uninstall(model)
    engine.close()
