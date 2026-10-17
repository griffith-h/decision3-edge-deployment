"""Official HTTP handling, serial inference on the model initialization thread."""
from concurrent.futures import Future
import queue
import threading

def serve_on_main_thread(model, port):
    from d3_server import serve
    pending = queue.Queue()
    server = serve(model, '0.0.0.0', port)
    answer = server.decider.answer
    def queued(body):
        future = Future()
        pending.put((body, future))
        return future.result()
    server.decider.answer = queued
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print('TEMPORARY_HTTP_READY', port, flush=True)
    try:
        while True:
            body, future = pending.get()
            try:
                future.set_result(answer(body))
            except Exception as exc:
                future.set_exception(exc)
            finally:
                pending.task_done()
    finally:
        server.shutdown()
        server.server_close()
