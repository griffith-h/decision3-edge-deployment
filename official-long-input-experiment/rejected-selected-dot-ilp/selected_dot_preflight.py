"""Observe old-model real dots; new kernels only write private comparison buffers."""
import hashlib,json,statistics,time
import torch
from selected_dot_ilp import SelectedDot

def run(model,owner,request,emit,out):
    state=owner.medium_sparse.repair;original=state.blas_repair;old_run=original.run
    kernels={width:SelectedDot(width) for width in (4,8,16)};records=[];benchmarks={};completed=False
    def save(response=None):
        data={'completed':completed,'scope':'Unchanged old model output; all actual selected dots checked on private buffers. Component timings are not model speed.',
              'records':records,'benchmarks':list(benchmarks.values()),'response':response,
              'mask_untouched':True,'k_order':'Each output retains continuous K0..K-1 FMA chain and final +0; only independent outputs interleaved'}
        (out/'selected-dot-preflight.json').write_text(json.dumps(data,indent=2)+'\n')
    def private(output,positions):
        value=output.clone();value[positions[:,0],positions[:,1]]=float('nan');return value
    def call(module,inputs,positions,output):
        index=len(records);old_run(module,inputs,positions,output)
        key=tuple(inputs.shape)+(module.weight.shape[0],);selected=len(positions)
        row={'index':index,'shape':list(key),'selected':selected,
             'positions_sha256':hashlib.sha256(positions.contiguous().numpy().tobytes()).hexdigest(),'variants':[]}
        records.append(row)
        try:
            raw4=None
            for width,kernel in kernels.items():
                actual=private(output,positions);raw=torch.empty(selected,dtype=torch.float32)
                kernel.run(module,inputs,positions,actual,raw)
                bf16_same=torch.equal(actual,output)
                fp32_same=raw4 is None or torch.equal(raw.view(torch.int32),raw4.view(torch.int32))
                row['variants'].append({'width':width,'whole_output_equal_old':bf16_same,'selected_fp32_bits_equal_width4_original_order':fp32_same})
                if raw4 is None:raw4=raw
                assert bf16_same and fp32_same,('Real selected-dot difference',row)
            if key not in benchmarks and selected:
                bench={'shape':list(key),'selected':selected,'pairs':[]};benchmarks[key]=bench
                for width in (8,16):
                    times=[]
                    for pair in range(3):
                        values={}
                        for mode in (('old','new') if pair%2==0 else ('new','old')):
                            actual=private(output,positions);begin=time.perf_counter()
                            (old_run if mode=='old' else kernels[width].run)(module,inputs,positions,actual)
                            elapsed=time.perf_counter()-begin
                            assert torch.equal(actual,output)
                            values[mode]=elapsed
                        times.append({'pair':pair,**values,'ratio':values['new']/values['old']})
                    bench['pairs'].append({'width':width,'records':times,'median_ratio':statistics.median(r['ratio'] for r in times),
                        'median_old':statistics.median(r['old'] for r in times),'median_new':statistics.median(r['new'] for r in times)})
            if len(records)%32==0:save()
        except BaseException:save();raise
    original.run=call
    try:
        emit({'event':'ready','case':'selected-dot-preflight'})
        emit({'event':'request_begin','phase':'unchanged_model_with_private_dot_comparisons','budget_seconds':120})
        begin=time.perf_counter();response=model.system_one(**request);elapsed=time.perf_counter()-begin
        completed=True;save(response)
        emit({'event':'complete','actual_dot_calls':len(records),'geometries':len(benchmarks),'instrumented_seconds':elapsed,'not_a_latency_benchmark':True})
    finally:original.run=old_run
