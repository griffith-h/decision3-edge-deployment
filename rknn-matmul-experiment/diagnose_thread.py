"""Measure the same warmed model on main/worker threads and flush-denormal modes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from pathlib import Path
import torch
from cpu_optimized_server import optimize
from rknn_linear import Engine, install, uninstall

parser=argparse.ArgumentParser(); parser.add_argument('--model',required=True)
args=parser.parse_args()
sys.path.insert(0,args.model)
from d3_runtime import D3
torch.set_num_threads(4)
model=D3.from_pretrained(args.model,device='cpu',verify='full')
optimize(model)
engine=Engine()
single=next(x for x in json.loads(Path('/work/request-cases.json').read_text()) if x['id']=='single')
records=[]
def invoke():
    return model.system_one(state=single['state'],questions=single['questions'])
def measure(label,flush):
    supported=torch.set_flush_denormal(flush)
    times=[]
    for _ in range(2):
        start=time.perf_counter(); response=invoke(); times.append(time.perf_counter()-start)
    r={'thread':label,'flush_denormal':flush,'supported':supported,'threads':torch.get_num_threads(),'seconds':times,'response':response}
    print(json.dumps(r,ensure_ascii=False),flush=True)
    return r
try:
    install(model,engine)
    invoke()
    for mode in [False,True]:
        records.append(measure('main',mode))
    with ThreadPoolExecutor(max_workers=1,initializer=lambda:torch.set_num_threads(4)) as worker:
        for mode in [False,True]:
            records.append(worker.submit(measure,'worker',mode).result())
    Path('/work/logs/thread-diagnosis.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
finally:
    uninstall(model); engine.close()
