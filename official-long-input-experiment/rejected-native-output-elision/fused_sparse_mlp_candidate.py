"""Restore existing dual-core native MLP fusion around sparse BF16 repair."""
import ctypes
from concurrent.futures import wait
import torch

def install(owner):
    state=owner.medium_sparse;native=owner.native_mlp;previous=native.run;lib=state.raw_native
    lib.d3_native_product_bf16.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]*2
    lib.d3_native_product_bf16.restype=ctypes.c_int
    state.counts['fused_sparse_mlp_chunks']=0
    def run(members,x):
        if not state.qualified() or not owner.long_mlp.active or tuple(x.shape[:2])!=(1,615):return previous(members,x)
        gate,up,down=members
        compute=getattr(state,'native_compute_no_output',None)
        if compute is None:native.group((gate,up),x)
        else:state.native_group_no_output((gate,up),x)
        gv=state.correct_native(gate,x);uv=state.correct_native(up,x)
        product=torch.empty_like(gv);parts=((0,320),(320,295));handles=[native.ensure_down(down,core) for core in (1,2)]
        outputs=[torch.empty((rows,down.out_features),dtype=torch.bfloat16) for _,rows in parts] if compute is None else [None,None]
        for handle,(start,rows) in zip(handles,parts):
            rc=lib.d3_native_product_bf16(handle,gv.data_ptr(),uv.data_ptr(),start,rows,615,down.in_features,native.table.silu.data_ptr(),product.data_ptr())
            if rc in (-2,-3):raise ValueError('Corrected MLP product finite/range guard')
            if rc:raise RuntimeError('Corrected native MLP product contract failed: '+str(rc))
        jobs=[]
        try:
            for handle,(_,rows),out in zip(handles,parts,outputs):
                jobs.append(owner.engine.pool.submit(lambda h=handle,n=rows,v=out:native.status(native.lib.nm_output(h,n,v.data_ptr()) if compute is None else compute(h,n))))
        finally:wait(jobs)
        for job in jobs:job.result()
        if compute is not None:state.counts['elided_native_outputs']+=len(handles)
        values=[]
        for handle,(start,rows) in zip(handles,parts):
            values.append(state.correct_native(down,product[:,start:start+rows],handle=handle))
        native.calls+=1;state.counts['mlp_chunks']+=1;state.counts['fused_sparse_mlp_chunks']+=1
        return torch.cat(values,dim=1)
    native.run=run
    def close():
        if native.run is run:native.run=previous
    return close
