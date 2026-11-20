"""Predeclared 2x2 cache-layout control, frozen kernels and captured real points."""
import hashlib,json,math,os,resource,statistics,time
from pathlib import Path
from types import SimpleNamespace
import torch
from enabled_sequential_repair_candidate import EnabledSequentialRepair,original_backend
from selected_dot_ilp import SelectedDot
from selected_dot_diagnosis import PmuClient

def run(emit,out,operating_state):
    torch.set_num_threads(3)
    assert original_backend() and not torch.is_autocast_enabled('cpu')
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/lab').iterdir() if p.is_file()}
    (out/'lab-sources.json').write_text(json.dumps(sources)+'\n')
    modes={'old':EnabledSequentialRepair().run,**{str(w):SelectedDot(w).run for w in (4,8,16)}}
    records=[];captures=[];completed=False;client=None;workloads=[]
    def save():
        (out/'cache-control.json').write_text(json.dumps({'completed':completed,'records':records,'captures':captures,
            'scope':'2 strides x2 base alignments, frozen original/4/8/16. CPU replay only; no engine change or model latency result.'})+'\n')
    def condition():
        state=operating_state()
        for p in Path('/sys/class/devfreq').glob('*/cur_freq'):
            try:state[str(p)]=p.read_text().strip()
            except OSError:pass
        return state
    with torch.inference_mode():
        for index in (10,15):
            file=Path('/replay')/f'case-{index}.pt';data=torch.load(file,weights_only=True,map_location='cpu')
            x=data['inputs'];positions=data['positions'];expected=data['expected'];weight=data['weight'];n,k=weight.shape
            captures.append({'index':index,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'shape':[len(x),k,n],
                             'selected':len(positions),'positions_sha256':hashlib.sha256(positions.numpy().tobytes()).hexdigest()})
            items=[]
            for padding in (0,32):
                for alignment in (0,16):
                    stride=k+padding;storage=torch.empty(n*stride+64,dtype=torch.bfloat16)
                    assert storage.data_ptr()%64==0
                    w=torch.as_strided(storage,(n,k),(stride,1),storage_offset=alignment//2);w.copy_(weight)
                    assert w.data_ptr()%64==alignment and torch.equal(w.view(torch.int16),weight.view(torch.int16))
                    module=SimpleNamespace(weight=w);buffers={}
                    for mode,fn in modes.items():
                        y=expected.clone();y[positions[:,0],positions[:,1]]=float('nan');fn(module,x,positions,y)
                        assert torch.equal(y.view(torch.int16),expected.view(torch.int16));buffers[mode]=y
                    items.append({'padding_bytes':padding*2,'alignment64':alignment,'module':module,'buffers':buffers})
            times=[]
            for _ in range(3):
                start=time.perf_counter();modes['old'](items[0]['module'],x,positions,items[0]['buffers']['old']);times.append(time.perf_counter()-start)
            repeats=max(4,min(32,math.ceil(.15/statistics.median(times))))
            workloads.append((captures[-1],x,positions,expected,items,repeats))
        try:
            client=PmuClient();emit({'event':'ready','case':'selected-dot-diagnosis-control'})
            for cap,x,positions,expected,items,repeats in workloads:
                for suite in ('core','cache'):
                    for round_number in range(8):
                        emit({'event':'request_begin','phase':'factorial_control','shape':cap['shape'],'suite':suite,'round':round_number,'budget_seconds':120})
                        # Each layout and each mode occupies every temporal position twice.
                        ordered=items[round_number%4:]+items[:round_number%4]
                        for item in ordered:
                            order=list(modes);rotation=(round_number+(item['padding_bytes']//64)*2+item['alignment64']//16)%4
                            order=order[rotation:]+order[:rotation]
                            for mode in order:
                                state=condition();usage=resource.getrusage(resource.RUSAGE_SELF);client.call(op='begin',suite=suite);start=time.perf_counter()
                                try:
                                    for _ in range(repeats):modes[mode](item['module'],x,positions,item['buffers'][mode])
                                    elapsed=time.perf_counter()-start
                                finally:pmu=client.call(op='end')
                                after=resource.getrusage(resource.RUSAGE_SELF)
                                records.append({'shape':cap['shape'],'capture_index':cap['index'],'suite':suite,'round':round_number,
                                    'padding_bytes':item['padding_bytes'],'alignment64':item['alignment64'],'mode':mode,'repeats':repeats,
                                    'selected':len(positions),'seconds_per_call':elapsed/repeats,'pmu':pmu,'state_before':state,'state_after':condition(),
                                    'voluntary_context_switches':after.ru_nvcsw-usage.ru_nvcsw,'involuntary_context_switches':after.ru_nivcsw-usage.ru_nivcsw})
                                assert not pmu['thread_set_changed'] and pmu['min_running_fraction']>=.99
                                assert torch.equal(item['buffers'][mode].view(torch.int16),expected.view(torch.int16))
                        save()
                    emit({'event':'factorial_suite_complete','shape':cap['shape'],'suite':suite})
            completed=True;save();emit({'event':'complete','pmu_windows':len(records),'all_bf16_bits_equal':True})
        finally:
            save()
            if client:client.close()
