"""Profile one actual original-CPU scan without changing its inputs or math."""
import copy,json,time
import torch

def run(model,owner,request,get_counts,emit,out):
    original=owner.long.original_scan;profiled=False;calls=0
    def forward(*args,**kwargs):
        nonlocal profiled,calls
        calls+=1
        if profiled:return original(*args,**kwargs)
        profiled=True
        metadata={'callable_module':getattr(original,'__module__',None),'callable_name':getattr(original,'__qualname__',None),
            'code_file':getattr(getattr(original,'__code__',None),'co_filename',None),
            'tensor_shapes':[list(a.shape) for a in args if isinstance(a,torch.Tensor)],
            'mkldnn':torch.backends.mkldnn.enabled,'threads':torch.get_num_threads(),
            'float32_precision':torch.get_float32_matmul_precision(),'inference_mode':torch.is_inference_mode_enabled()}
        begin=time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True) as profiler:
            result=original(*args,**kwargs)
        metadata['instrumented_wall_seconds']=time.perf_counter()-begin
        rows=[{'operator':e.key,'calls':e.count,'self_cpu_us':e.self_cpu_time_total,'inclusive_cpu_us':e.cpu_time_total,'input_shapes':e.input_shapes}
              for e in profiler.key_averages(group_by_input_shape=True)]
        rows.sort(key=lambda r:r['self_cpu_us'],reverse=True)
        (out/'scan-operators.json').write_text(json.dumps({'completed':True,'scope':'First actual original CPU scan only; profiler overhead included; not model latency benchmark','metadata':metadata,'operators':rows,'inclusive_times_overlap':True},indent=2)+'\n')
        return result
    owner.long.original_scan=forward;before=copy.deepcopy(get_counts())
    emit({'event':'ready','case':'scan-operator-profile'})
    try:
        emit({'event':'request_begin','phase':'full_diagnostic_with_one_scan_profile','budget_seconds':120})
        begin=time.perf_counter();response=model.system_one(**request);elapsed=time.perf_counter()-begin
        if not profiled:raise AssertionError('Original scan profiling hook did not run')
        (out/'scan-profile-result.json').write_text(json.dumps({'completed':True,'response':response,'instrumented_seconds':elapsed,
            'original_scan_calls':calls,'profiled_scan_calls':1,'counts_before':before,'counts_after':get_counts(),
            'medium_sparse_counts':owner.medium_sparse.snapshot(),'long_mlp_counts':owner.long_mlp.counts})+'\n')
        emit({'event':'complete','original_scan_calls':calls,'profiled_scan_calls':1,'instrumented_seconds':elapsed})
    finally:
        if owner.long.original_scan is forward:owner.long.original_scan=original
