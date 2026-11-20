"""Skip only discarded BF16 MLP outputs; same SDK runs and old range guards."""
import ctypes
from concurrent.futures import wait

def install(owner):
    state=owner.medium_sparse;native=owner.native_mlp
    if 'fused_sparse_mlp_chunks' not in state.counts:raise RuntimeError('Fused sparse MLP required')
    compute=state.raw_native.d3_native_run_checked
    compute.argtypes=[ctypes.c_void_p,ctypes.c_int];compute.restype=ctypes.c_int
    missing=object();old_compute=getattr(state,'native_compute_no_output',missing);old_group=getattr(state,'native_group_no_output',missing)
    old_count=state.counts.get('elided_native_outputs',missing);state.counts['elided_native_outputs']=0;closed=False
    def group(members,x):
        rows=x.numel()//x.shape[-1];flat=x.reshape(-1,x.shape[-1]).contiguous()
        handles={id(m):native.ensure(m) for m in members}
        native.status(native.lib.nm_prepare(handles[id(members[0])],rows,flat.data_ptr()))
        for m in members[1:]:native.status(native.lib.nm_copy(handles[id(m)],handles[id(members[0])],rows))
        buckets=[[m for m in members if owner.engine.bucket.get(id(m),0)==i] for i in range(2)]
        jobs=[]
        def execute(items):
            for module in items:native.status(compute(handles[id(module)],rows))
        try:
            for bucket in buckets:
                if bucket:jobs.append(owner.engine.pool.submit(execute,bucket))
        finally:wait(jobs)
        for job in jobs:job.result()
        state.counts['elided_native_outputs']+=len(members)
    state.native_compute_no_output=compute;state.native_group_no_output=group
    def close():
        nonlocal closed
        if closed:return
        closed=True
        for name,previous in (('native_compute_no_output',old_compute),('native_group_no_output',old_group)):
            if previous is missing:delattr(state,name)
            else:setattr(state,name,previous)
        if old_count is missing:del state.counts['elided_native_outputs']
        else:state.counts['elided_native_outputs']=old_count
    return close
