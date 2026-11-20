"""Frozen-kernel PMU diagnosis, including one cache-conflict falsification control.

No engine kernel is changed. Padded copies change only the diagnostic weight row
stride by one64-byte line; selected indices, values and per-output K order stay.
"""
import copy,hashlib,json,math,socket,statistics,time
from pathlib import Path
from types import SimpleNamespace
import torch
from selected_dot_ilp import SelectedDot

class PmuClient:
    def __init__(self):
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.socket.settimeout(30)
        self.socket.connect('/pmu/monitor.sock');self.stream=self.socket.makefile('rwb',buffering=0)
    def call(self,**request):
        self.stream.write((json.dumps(request)+'\n').encode());line=self.stream.readline(262144)
        if not line:raise RuntimeError('Host PMU observer disconnected')
        value=json.loads(line)
        if not value.get('ok'):raise RuntimeError('PMU observation failed')
        return value
    def close(self):
        try:self.call(op='quit')
        finally:self.stream.close();self.socket.close()

def run(model,owner,request,emit,out,operating_state):
    state=owner.medium_sparse.repair;original=state.blas_repair;old_run=original.run
    kernels={str(w):SelectedDot(w) for w in (4,8,16)}
    captures={};calls=[];records=[];client=None;response=None;completed=False
    def save():
        (out/'selected-dot-diagnosis.json').write_text(json.dumps({'completed':completed,'response':response,
            'capture_calls':calls,'captures':[{k:v for k,v in cap.items() if k not in ('module','inputs','positions','expected')} for cap in captures.values()],
            'records':records,'scope':'Unchanged frozen kernels on same real selected points. PMU thread sums are not model wall time; pad64 is a diagnostic layout counterfactual only.'},indent=2)+'\n')
    def observe(module,inputs,positions,output):
        old_run(module,inputs,positions,output);key=tuple(inputs.shape)+(module.weight.shape[0],)
        call={'index':len(calls),'shape':list(key),'selected':len(positions),
              'positions_sha256':hashlib.sha256(positions.contiguous().numpy().tobytes()).hexdigest()};calls.append(call)
        if key in captures:return
        x=torch.empty_strided(inputs.shape,inputs.stride(),dtype=inputs.dtype);x.copy_(inputs)
        captures[key]={**call,'module':module,'inputs':x,'positions':positions.clone(),'expected':output.clone(),
                       'input_stride':list(inputs.stride()),'weight_stride':list(module.weight.stride()),
                       'input_alignment64':inputs.data_ptr()%64,'replay_input_alignment64':x.data_ptr()%64,
                       'weight_alignment64':module.weight.data_ptr()%64}
    original.run=observe;emit({'event':'ready','case':'selected-dot-diagnosis'})
    try:
        emit({'event':'request_begin','phase':'unchanged_model_capture','budget_seconds':120});start=time.perf_counter()
        response=model.system_one(**request);capture_seconds=time.perf_counter()-start
    finally:original.run=old_run
    emit({'event':'capture_complete','seconds':capture_seconds,'calls':len(calls),'geometries':len(captures)})
    save()
    try:
        with torch.inference_mode():
            workloads=[]
            for key,cap in captures.items():
                cap['capture_file']='case-'+str(cap['index'])+'.pt'
                # Sensitive tensors stay in the board-only private capture mount.
                target=Path('/captures')/cap['capture_file']
                torch.save({'inputs':cap['inputs'],'weight':cap['module'].weight.detach().clone(),
                            'positions':cap['positions'],'expected':cap['expected']},target)
                cap['capture_file_sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
                workloads.append((key,'original',cap['module'],cap))
                if key in ((512,2048,1024),(615,1024,3584)):
                    n,k=cap['module'].weight.shape
                    padded=torch.empty_strided((n,k),(k+32,1),dtype=torch.bfloat16);padded.copy_(cap['module'].weight)
                    assert torch.equal(padded.view(torch.int16),cap['module'].weight.view(torch.int16))
                    workloads.append((key,'pad64',SimpleNamespace(weight=padded),cap))
            modes={'old':old_run,**{width:kernel.run for width,kernel in kernels.items()}}
            prepared=[]
            for key,layout,module,cap in workloads:
                x=cap['inputs'];positions=cap['positions'];expected=cap['expected'];buffers={}
                for name,fn in modes.items():
                    value=expected.clone();value[positions[:,0],positions[:,1]]=float('nan')
                    fn(module,x,positions,value)
                    assert torch.equal(value.view(torch.int16),expected.view(torch.int16)),('Frozen kernel bit difference',key,layout,name)
                    buffers[name]=value
                timings=[]
                for _ in range(3):
                    start=time.perf_counter();old_run(module,x,positions,buffers['old']);timings.append(time.perf_counter()-start)
                repeats=max(4,min(32,math.ceil(.15/statistics.median(timings))))
                prepared.append((key,layout,module,cap,buffers,repeats))
            client=PmuClient()
            for key,layout,module,cap,buffers,repeats in prepared:
                for suite in ('core','cache','front'):
                    emit({'event':'request_begin','phase':'pmu_replay','shape':list(key),'layout':layout,'suite':suite,'budget_seconds':120})
                    for repeat in range(4):
                        order=['old','4','8','16'];order=order[repeat:]+order[:repeat]
                        for name in order:
                            condition=operating_state();client.call(op='begin',suite=suite)
                            start=time.perf_counter()
                            try:
                                for _ in range(repeats):modes[name](module,cap['inputs'],cap['positions'],buffers[name])
                                elapsed=time.perf_counter()-start
                            finally:pmu=client.call(op='end')
                            record={'shape':list(key),'capture_index':cap['index'],'layout':layout,'weight_stride':list(module.weight.stride()),
                                'mode':name,'suite':suite,'round':repeat,'repeats':repeats,'selected':cap['selected'],'seconds':elapsed,
                                'seconds_per_call':elapsed/repeats,'pmu':pmu,'operating_state_before':condition,'operating_state_after':operating_state()}
                            records.append(record);save()
                            assert not pmu['thread_set_changed'] and pmu['min_running_fraction']>=.99,'PMU threads changed or events multiplexed'
                            assert pmu['totals']['cpu_cycles']>0 and pmu['totals']['inst_retired']>0
                            assert torch.equal(buffers[name].view(torch.int16),cap['expected'].view(torch.int16))
                    emit({'event':'pmu_suite_complete','shape':list(key),'layout':layout,'suite':suite})
            completed=True;save();emit({'event':'complete','geometries':len(captures),'workloads':len(prepared),'pmu_windows':len(records),'not_a_model_latency_benchmark':True})
    finally:
        save()
        if client:client.close()
