"""HTTP service for the evaluated native-BF16 Owner, inference on its main thread."""
import argparse
from concurrent.futures import Future
import copy
import json
import os
from pathlib import Path
import queue
import signal
import threading
import time

from board import setup
from evaluation_verdict import require_final_complete

RELEASE = '2026-10-19-native-bf16-v1'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=8900)
    p.add_argument('--evaluation', default='/evaluation/evaluation-result.json')
    a = p.parse_args()
    verdict = json.loads(Path(a.evaluation).read_text())
    require_final_complete(verdict)
    if os.environ.get('OMP_NUM_THREADS') != '3' or os.environ.get('OMP_THREAD_LIMIT') != '4':
        raise RuntimeError('Evaluated Torch3 / scan4 configuration required')
    model = close = get_counts = server = worker = None
    stopping = threading.Event()
    admission = threading.Lock()
    pending = queue.Queue()
    snapshot = {'requests_completed': 0, 'last_request': None, 'counts': {}}

    def stop(signum, frame):
        stopping.set()
        pending.put(None)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        model, close, get_counts = setup('native-bf16')
        from d3_server import serve
        body = json.loads(Path('/service/request.json').read_text())
        long_body = dict(body, state=' '.join(['alpha'] * 512))
        for item in (long_body, body):
            result = model.system_one(state=item['state'], questions=item['questions'])
            if any('error' in answer for answer in result['answers'].values()):
                raise RuntimeError('Warmup returned an invalid model answer')
        snapshot['counts'] = copy.deepcopy(get_counts())
        server = serve(model, '0.0.0.0', a.port, api_key=os.environ.get('DECISION_API_KEY') or None)
        original_answer, original_health = server.decider.answer, server.decider.health

        def health(authentication):
            result = original_health(authentication)
            result['deployment'] = {'release': RELEASE, 'final_stage': 'native-bf16',
                                    'sample_rows_sha256': verdict['sample_rows_sha256'],
                                    'torch_threads': 3, 'OMP_THREAD_LIMIT': 4,
                                    'affinity': sorted(os.sched_getaffinity(0)),
                                    'capacity': {'max_length': model.max_length, 'batch_size': model.batch_size},
                                    'scope': 'native BF16 MLP only B1/T615; other shapes retain historical guarded paths',
                                    **snapshot}
            return result

        def queued_answer(body):
            future = Future()
            with admission:
                if stopping.is_set():
                    raise RuntimeError('Service is stopping')
                pending.put((body, future))
            return future.result()

        server.decider.answer, server.decider.health = queued_answer, health
        worker = threading.Thread(target=server.serve_forever, name='d3-http', daemon=True)
        worker.start()
        print('SERVICE_READY', RELEASE, flush=True)
        while not stopping.is_set():
            job = pending.get()
            if job is None:
                break
            body, future = job
            try:
                start = time.perf_counter()
                result = original_answer(body)
                elapsed = time.perf_counter() - start
                snapshot = {'requests_completed': snapshot['requests_completed'] + 1,
                            'last_request': {'seconds': elapsed, 'input_tokens': result.get('usage', {}).get('input_tokens')},
                            'counts': copy.deepcopy(get_counts())}
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)
    finally:
        stopping.set()
        with admission:
            while not pending.empty():
                job = pending.get_nowait()
                if job is not None:
                    job[1].set_exception(RuntimeError('Service is stopping'))
        if server:
            if worker:
                server.shutdown()
                worker.join(timeout=10)
            server.server_close()
        if close:
            close()
        print('SERVICE_STOPPED_CLEANLY', flush=True)


if __name__ == '__main__':
    main()
