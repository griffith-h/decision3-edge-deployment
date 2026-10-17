"""Production entry point for the validated three-thread RKNN hybrid release."""
import argparse
from concurrent.futures import Future
import json
import os
from pathlib import Path
import queue
import signal
import sys
import threading
import torch

RELEASE = '2026-10-17-three-thread-v1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=8900)
    args = parser.parse_args()
    sys.path[:0] = ['/base', args.model]
    from cpu_optimized_server import optimize
    from rknn_linear import Engine, install, uninstall
    from cpu_attention_optimized import install_conv, install_scan, install_sdpa
    from native_norm import install_norm, install_native_conv
    from d3_runtime import D3, PromptEncoder
    from d3_server import serve
    assert os.environ.get('OMP_THREAD_LIMIT') == '3'
    assert os.environ.get('OMP_NUM_THREADS') == '3'
    torch.set_num_threads(3)
    model = D3.from_pretrained(args.model, device='cpu', verify='full')
    model.model_name = 'vllm-sr/d3-edge'
    assert model.attention_mode == 'noncausal_full_attention'
    assert isinstance(model.encoder, PromptEncoder) and len(model.encoder.stack) == 12
    optimize(model)
    engine = Engine()
    restore, stats = [], {}
    server, http_thread = None, None
    pending = queue.Queue()
    stopping = threading.Event()
    admission = threading.Lock()
    def stop(signum, frame):
        stopping.set()
        pending.put(None)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        selected = install(model, engine)
        assert len(selected) == 78
        restore.extend([install_conv(), install_scan(128), install_sdpa(),
                        install_norm(model.backbone, stats), install_native_conv(stats)])
        body = json.loads(Path('/base/request.json').read_text())
        model.system_one(state=body['state'], questions=body['questions'])
        assert stats.get('norm_calls', 0) > 0 and stats.get('conv_calls', 0) > 0
        if stopping.is_set():
            return
        server = serve(model, '0.0.0.0', args.port,
                       api_key=os.environ.get('DECISION_API_KEY') or None)
        original_answer, original_health = server.decider.answer, server.decider.health
        def health(authentication):
            result = original_health(authentication)
            result['deployment'] = {'release': RELEASE, 'torch_threads': torch.get_num_threads(),
                'OMP_THREAD_LIMIT': os.environ['OMP_THREAD_LIMIT'],
                'affinity': sorted(os.sched_getaffinity(0)), 'chunk_size': 128,
                'rknn_linear_layers': len(selected), 'native_stats': dict(stats)}
            return result
        def queued_answer(body):
            future = Future()
            with admission:
                if stopping.is_set():
                    raise RuntimeError('Service is stopping')
                pending.put((body, future))
            return future.result()
        server.decider.answer, server.decider.health = queued_answer, health
        http_thread = threading.Thread(target=server.serve_forever, name='d3-http', daemon=True)
        http_thread.start()
        print('SERVICE_READY', json.dumps(health(False)), flush=True)
        while not stopping.is_set():
            item = pending.get()
            if item is None:
                break
            body, future = item
            try:
                future.set_result(original_answer(body))
            except Exception as exc:
                future.set_exception(exc)
    finally:
        stopping.set()
        with admission:
            while not pending.empty():
                item = pending.get_nowait()
                if item is not None:
                    item[1].set_exception(RuntimeError('Service is stopping'))
        if server is not None:
            if http_thread is not None:
                server.shutdown()
            server.server_close()
        for callback in reversed(restore):
            callback()
        uninstall(model)
        engine.close()
        print('SERVICE_STOPPED_CLEANLY', flush=True)

if __name__ == '__main__':
    main()
