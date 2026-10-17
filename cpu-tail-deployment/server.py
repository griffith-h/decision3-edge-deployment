"""Main-thread serialized HTTP owner for the frozen graph-half RKNN candidate."""
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
import time

RELEASE = '2026-10-18-graph-half-v1'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=8900)
    args = parser.parse_args()
    sys.path[:0] = ['/work', '/service', args.model]
    from runtime import Runtime
    from d3_server import serve
    assert os.environ.get('OMP_THREAD_LIMIT') == '3'
    assert os.environ.get('OMP_NUM_THREADS') == '3'
    torch.set_num_threads(3)
    runtime = None
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
        runtime = Runtime()
        model = runtime.model
        body = json.loads(Path('/service/request.json').read_text())
        model.system_one(state=body['state'], questions=body['questions'])
        assert runtime.graph_counts['optimized'] == 12
        assert not runtime.graph_counts['errors']
        if stopping.is_set():
            return
        server = serve(model, '0.0.0.0', args.port,
                       api_key=os.environ.get('DECISION_API_KEY') or None)
        completed, last_request = [0], [None]
        original_answer, original_health = server.decider.answer, server.decider.health
        def health(authentication):
            result = original_health(authentication)
            result['deployment'] = {'release': RELEASE, 'torch_threads': torch.get_num_threads(),
                'OMP_THREAD_LIMIT': os.environ['OMP_THREAD_LIMIT'],
                'affinity': sorted(os.sched_getaffinity(0)),
                'graph_signature': [1, 115, 1024],
                'other_shapes': 'CPU fused fallback', 'counts': runtime.counts(),
                'requests_completed': completed[0], 'last_request': last_request[0]}
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
                before = runtime.graph_counts['optimized']
                fallbacks = runtime.graph_counts['fallback']
                started = time.perf_counter()
                result = original_answer(body)
                last_request[0] = {'seconds': time.perf_counter()-started,
                    'graph_calls': runtime.graph_counts['optimized']-before,
                    'graph_fallbacks': runtime.graph_counts['fallback']-fallbacks,
                    'input_tokens': result.get('usage', {}).get('input_tokens')}
                completed[0] += 1
                future.set_result(result)
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
        if runtime is not None:
            runtime.close()
        print('SERVICE_STOPPED_CLEANLY', flush=True)

if __name__ == '__main__':
    main()
