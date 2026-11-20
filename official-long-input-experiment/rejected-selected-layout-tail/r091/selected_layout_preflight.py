"""Real selected-point parity and fixed causal ablations; old model output retained."""
import copy,hashlib,json,statistics,time
import torch
from selected_dot_ilp import SelectedDot
from selected_layout_tail import LayoutTail

def run(model,owner,request,emit,out,memory_snapshot):
    original=owner.medium_sparse.repair.blas_repair;old_run=original.run
    trusted=lambda module:any(module is item for modules,_ in owner.engine.packed.values() for item in modules)
    kernels={'layout_only':LayoutTail(4,True,trusted_inference=trusted),'tail_only':LayoutTail(16,False,trusted_inference=trusted),'combined':LayoutTail(16,True,trusted_inference=trusted)}
    reference=SelectedDot(4);records=[];benchmarks={};weights={};completed=False;response=None
    before=memory_snapshot()
    def save():
        (out/'layout-preflight.json').write_text(json.dumps({'completed':completed,'records':records,'benchmarks':list(benchmarks.values()),'response':response,
            'caches':{name:{'counts':dict(kernel.counts),'live_bytes':kernel.live_bytes,'entries':len(kernel.cache)} for name,kernel in kernels.items()},
            'unique_weight_shapes':list(weights.values()),'memory_before':before,'memory_after':memory_snapshot(),'scope':'Old model output with private parity/ablation replays; not candidate model latency.'},indent=2)+'\n')
    def observe(module,inputs,positions,output):
        old_run(module,inputs,positions,output);key=tuple(inputs.shape)+(module.weight.shape[0],);selected=len(positions)
        weights[id(module)]={'shape':list(module.weight.shape),'bf16_bytes':module.weight.numel()*2}
        row={'index':len(records),'shape':list(key),'selected':selected,'positions_sha256':hashlib.sha256(positions.contiguous().numpy().tobytes()).hexdigest(),'modes':{}};records.append(row)
        raw=torch.empty(selected,dtype=torch.float32);check=output.clone();reference.run(module,inputs,positions,check,raw)
        assert torch.equal(check.view(torch.int16),output.view(torch.int16))
        for name,kernel in kernels.items():
            actual=output.clone();actual[positions[:,0],positions[:,1]]=float('nan');fp=torch.empty_like(raw)
            kernel.run(module,inputs,positions,actual,fp)
            row['modes'][name]={'bf16_bits_equal':torch.equal(actual.view(torch.int16),output.view(torch.int16)),'fp32_bits_equal':torch.equal(fp.view(torch.int32),raw.view(torch.int32))}
            assert all(row['modes'][name].values()),row
        if key not in benchmarks and selected:
            b={'shape':list(key),'selected':selected,'modes':{}};benchmarks[key]=b
            for name,kernel in kernels.items():
                pairs=[]
                for pair in range(3):
                    values={}
                    for mode in (('old','new') if pair%2==0 else ('new','old')):
                        actual=output.clone();begin=time.perf_counter();(old_run if mode=='old' else kernel.run)(module,inputs,positions,actual);values[mode]=time.perf_counter()-begin
                        assert torch.equal(actual.view(torch.int16),output.view(torch.int16))
                    pairs.append(values)
                b['modes'][name]={'pairs':pairs,'median_old':statistics.median(r['old'] for r in pairs),'median_new':statistics.median(r['new'] for r in pairs)}
        if len(records)%32==0:save()
    for kernel in kernels.values():kernel.begin()
    original.run=observe
    try:
        emit({'event':'ready','case':'selected-layout-preflight'});emit({'event':'request_begin','phase':'old_model_private_layout_tail_validation','budget_seconds':120})
        start=time.perf_counter();response=model.system_one(**request);seconds=time.perf_counter()-start
        completed=True;emit({'event':'complete','actual_calls':len(records),'geometries':len(benchmarks),'instrumented_seconds':seconds,'not_candidate_latency':True})
    finally:
        original.run=old_run
        for kernel in kernels.values():kernel.close()
        save()
