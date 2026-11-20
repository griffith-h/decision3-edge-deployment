"""Serial JSON RPC on the model-owning main thread; no gold reaches inference."""
import argparse
import contextlib
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from stages import MODEL, stage_spec

FRAME = '@@D3EVAL@@'


def send(value):
    print(FRAME + json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def operating_state():
    paths = ['/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq',
             '/sys/class/devfreq/27700000.npu/cur_freq']
    paths += [str(p) for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')]
    result = {}
    for name in paths:
        try:
            result[name] = Path(name).read_text().strip()
        except OSError:
            pass
    return result


def setup(stage):
    import torch
    spec = stage_spec(stage)
    torch.set_num_threads(spec['threads'])
    restores = []
    counts = lambda: {}
    owner = engine = None
    cleanup = lambda: None
    sys.path[:0] = ['/work', '/release', '/base', MODEL]

    if stage in ('official-bf16', 'cpu-fp32', 'rknn-initial', 'cpu-three-thread',
                 'bridge-parallel', 'scan-native'):
        from d3_runtime import D3
        model = D3.from_pretrained(MODEL, device='cpu', verify='full')
        model.model_name = 'vllm-sr/d3-edge'
        if stage != 'official-bf16':
            from cpu_optimized_server import optimize
            optimize(model)
        if stage in ('rknn-initial', 'cpu-three-thread'):
            from rknn_linear import Engine, install, uninstall
            engine = Engine()
            install(model, engine)
            restores.append(lambda: uninstall(model))
        if stage in ('cpu-three-thread', 'bridge-parallel', 'scan-native'):
            from cpu_attention_optimized import install_conv, install_scan, install_sdpa
            from native_norm import install_norm, install_native_conv
            native = {}
            restores.extend([install_conv(), install_scan(128), install_sdpa(),
                             install_norm(model.backbone, native), install_native_conv(native)])
        if stage in ('bridge-parallel', 'scan-native'):
            sys.path.insert(0, '/combined')
            from adapter import Engine, install
            engine = Engine(fusion=True, parallel=True)
            restores.append(install(model, engine))
            if stage == 'scan-native':
                sys.path.insert(0, '/scan')
                from scan import make_scan
                from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
                old = qwen.torch_chunk_gated_delta_rule
                qwen.torch_chunk_gated_delta_rule = make_scan(old, tile=16, fused=5)
                restores.append(lambda: setattr(qwen, 'torch_chunk_gated_delta_rule', old))
        def counts():
            return {} if engine is None else {'calls': engine.calls, 'cpu_fallbacks': engine.cpu_fallbacks,
                                              'engine': getattr(engine, 'stats', {})}
        def cleanup():
            for restore in reversed(restores):
                restore()
            if engine is not None:
                engine.close()
    elif stage == 'safe-half':
        from common import load
        model, engine, native, cleanup = load(MODEL, safe_output=True)
        counts = lambda: {'calls': engine.calls, 'cpu_fallbacks': engine.cpu_fallbacks, 'engine': engine.stats}
    elif stage == 'projection-merge':
        import setup_helpers
        from common import load
        from adapter import install
        from merged_adapter import PackedEngine
        model, old, native, close = load(MODEL, safe_output=True)
        old.restore_forwards()
        old.restore_forwards = None
        engine = PackedEngine('merge')
        setup_helpers.configure(engine)
        engine.groups = []
        restore = install(model, engine)
        def cleanup():
            restore()
            engine.close()
            close()
        counts = lambda: {'calls': engine.calls, 'cpu_fallbacks': engine.cpu_fallbacks, 'engine': engine.stats}
    elif stage in ('layout-scatter', 'scan-blas2'):
        # These are the same adopted wrappers used by later baseline.py, without later CPU fusions.
        sys.path[:0] = ['/previous', '/attn', '/work']
        from loader import open_model
        from conv_btc import install as install_conv
        from prepared_scan import install as install_prepare
        model, engine, native, close = open_model(library='/previous/libconversion_bridge.so')
        rc, cc = install_conv()
        rp, cp = install_prepare()
        restores = [rc, rp]
        if stage == 'scan-blas2':
            from chunk_blas2 import install
            rs, cs = install(32, 64)
            restores.append(rs)
        def cleanup():
            for restore in reversed(restores):
                restore()
            close()
        counts = lambda: {'calls': engine.calls, 'cpu_fallbacks': engine.cpu_fallbacks, 'engine': engine.stats}
    elif stage == 'graph-half':
        sys.path[:0] = ['/work', '/service']
        from runtime import Runtime
        owner = Runtime()
        model, cleanup, counts = owner.model, owner.close, owner.counts
    else:
        sys.path[:0] = ['/experiment', '/tail', '/long', '/row', '/mlp', '/work']
        modules = {'mlp-length': 'experiment', 'row-projection': 'fused_owner',
                   'long-sequence': 'owner', 'long-tail': 'runner',
                   'fusion-precision': 'selected_owner', 'native-bf16': 'accepted_owner'}
        owner = importlib.import_module(modules[stage]).Owner()
        if stage == 'mlp-length':
            from planned_mlp import PlannedBank
            owner.bank.close()
            owner.bank = PlannedBank(owner.model)
        modes = {'mlp-length': 'planned', 'row-projection': 'fused_guarded',
                 'long-sequence': 'both', 'long-tail': 'validate_mlp',
                 'fusion-precision': 'candidate', 'native-bf16': 'candidate'}
        owner.use(modes[stage])
        model, cleanup = owner.model, owner.close
        counts = getattr(owner, 'counts', lambda: {'mlp': owner.bank.counts})
    assert len(model.encoder.stack) == 12
    assert model.attention_mode == 'noncausal_full_attention'
    # Inference itself is identical to the historical model method. Telemetry is outside timing.
    return model, cleanup, counts


def identity(stage):
    import torch
    files = {}
    for folder in ('/evaluation', '/work', '/experiment', '/release', '/base', '/references', '/combined',
                   '/scan', '/remainder', '/projection', '/previous', '/attn', '/mlp',
                   '/row', '/long', '/tail', '/fusion', '/native'):
        path = Path(folder)
        if not path.is_dir():
            continue
        for f in sorted(path.iterdir()):
            if f.is_file() and f.suffix in ('.py', '.c', '.cpp', '.inc', '.json', '.so', '.rknn'):
                h = hashlib.sha256()
                with f.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                        h.update(chunk)
                files[str(f)] = h.hexdigest()
    # This nested historical plan is read by the final long-input Owner.
    plan = Path('/tail/logs/formal.json')
    if plan.is_file():
        files[str(plan)] = hashlib.sha256(plan.read_bytes()).hexdigest()
    return {'stage': stage_spec(stage), 'torch': torch.__version__, 'threads': torch.get_num_threads(),
            'affinity': sorted(os.sched_getaffinity(0)), 'model': MODEL, 'files': files}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True)
    a = p.parse_args()
    model = close = counts = None
    try:
        # Python diagnostics go to stderr; framing also tolerates native-library stdout.
        with contextlib.redirect_stdout(sys.stderr):
            model, close, counts = setup(a.stage)
            # Same public self-built warmups for every stage; never use sampled labels.
            body = json.loads(Path('/service/request.json').read_text())
            for warmup in (dict(body, state=' '.join(['alpha'] * 512)), body):
                out = model.system_one(state=warmup['state'], questions=warmup['questions'])
                if any('error' in answer for answer in out['answers'].values()):
                    raise RuntimeError('Initial warmup returned an invalid model answer')
            provenance = identity(a.stage)
            provenance['runtime_capacity'] = {'max_length': model.max_length, 'batch_size': model.batch_size}
            provenance['initial_warmups'] = ['alpha512 / 615 tokens', 'frozen service request / 115 tokens']
        send({'event': 'ready', 'identity': provenance})
        for line in sys.stdin:
            request = json.loads(line)
            seq = request['seq']
            payload = request['payload']
            if set(payload) != {'state', 'questions'}:
                raise ValueError('Only state/questions may reach inference')
            condition = operating_state()
            before = copy.deepcopy(counts())
            start = time.perf_counter()
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    out = model.system_one(**payload)
                elapsed = time.perf_counter() - start
                send({'seq': seq, 'status': 'ok', 'response': out,
                      'board_method_ms': elapsed * 1000, 'counts_before': before,
                      'counts': counts(), 'operating_state': condition})
            except Exception as exc:
                send({'seq': seq, 'status': 'error', 'exception': type(exc).__name__,
                      'error': str(exc), 'board_method_ms': (time.perf_counter() - start) * 1000,
                      'traceback': traceback.format_exc()})
    finally:
        if close:
            with contextlib.redirect_stdout(sys.stderr):
                close()


if __name__ == '__main__':
    main()
