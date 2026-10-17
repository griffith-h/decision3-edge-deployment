#!/usr/bin/env python3
"""Separate experimental endpoint; reuse the official serialized D3 HTTP server."""
import argparse
from concurrent.futures import Future
import json
import os
import queue
import sys
import threading
from pathlib import Path
import torch
from cpu_optimized_server import optimize
from rknn_linear import Engine, install, uninstall

parser=argparse.ArgumentParser()
parser.add_argument('--model',required=True,type=Path)
parser.add_argument('--host',default='0.0.0.0')
parser.add_argument('--port',type=int,default=8900)
args=parser.parse_args()
sys.path.insert(0,str(args.model))
from d3_runtime import D3, PromptEncoder
from d3_server import serve
torch.set_num_threads(4)
model=D3.from_pretrained(args.model,device='cpu',verify='full')
model.model_name='vllm-sr/d3-edge'
assert model.attention_mode=='noncausal_full_attention' and isinstance(model.encoder,PromptEncoder)
optimize(model)
engine=Engine()
server=None
requests=queue.Queue()
http_thread=None
active=None
try:
    selected=install(model,engine)
    body=json.loads(Path('/work/request.json').read_text())
    model.system_one(state=body['state'],questions=body['questions'])
    print(f'RKNN warmup complete: {len(selected)} text Linear layers, {len(engine.handles)} cached NPU handles; original noncausal PromptEncoder',flush=True)
    # Decider.turn in the official server serializes model.run across requests.
    server=serve(model,args.host,args.port,api_key=os.environ.get('DECISION_API_KEY') or None)
    original_answer=server.decider.answer
    # The board measured significantly faster inference on the initialization
    # thread than Python auxiliary threads. HTTP keeps its original handlers;
    # model calls execute serially on this main thread and reuse native buffers.
    def queued_answer(body):
        future=Future()
        requests.put((body,future))
        return future.result()
    server.decider.answer=queued_answer
    print(f'Experimental CPU+RKNN service ready on {args.host}:{args.port}',flush=True)
    http_thread=threading.Thread(target=server.serve_forever,name='d3-http',daemon=True)
    http_thread.start()
    while True:
        body,active=requests.get()
        try:
            active.set_result(original_answer(body))
        except Exception as exc:
            active.set_exception(exc)
        finally:
            requests.task_done()
        active=None
finally:
    if server:
        if http_thread: server.shutdown()
        server.server_close()
    if active is not None and not active.done():
        active.set_exception(RuntimeError('Service is stopping'))
    while not requests.empty():
        _,pending=requests.get_nowait()
        pending.set_exception(RuntimeError('Service is stopping'))
    uninstall(model)
    engine.close()
