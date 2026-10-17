"""Balanced packing with direct native conversion into original output buffers."""
import ctypes,time
import torch
from merged_adapter import PackedEngine as TensorPackedEngine
from safe_engine import OutputRangeError

class PackedEngine(TensorPackedEngine):
    def __init__(self,mode):
        super().__init__(mode);self.plans={};self.profile=False;self.events=[]
        old=self.lib
        # Configure every entry point against the new library before any handles.
        self.lib=ctypes.CDLL('/work/libsafe_scatter_bridge.so')
        for name in ('mm_error','mm_create_core','mm_run','mm_destroy','mm_prepare_bf16','mm_copy_prepared','mm_run_prepared_bf16','mm_run_bf16'):
            src=getattr(old,name);dst=getattr(self.lib,name);dst.argtypes=src.argtypes;dst.restype=src.restype
        fn=self.lib.mm_run_prepared_scatter_bf16
        fn.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]+[ctypes.POINTER(ctypes.c_int)]*3+[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p];fn.restype=ctypes.c_int
    def plan(self,modules,packed):
        key=tuple(id(m) for m in modules)
        if key in self.plans:return self.plans[key]
        boundaries=[0]
        for m in modules:boundaries.append(boundaries[-1]+m.out_features)
        plans=[];offset=0
        for m in packed:
            segments=[];end=offset+m.out_features
            for index,original in enumerate(modules):
                lo=max(offset,boundaries[index]);hi=min(end,boundaries[index+1])
                if hi>lo:segments.append((lo-offset,hi-lo,original.out_features,id(original),lo-boundaries[index]))
            ints=ctypes.c_int*len(segments)
            plans.append((segments,ints(*[s[0] for s in segments]),ints(*[s[1] for s in segments]),ints(*[s[2] for s in segments])))
            offset=end
        self.plans[key]=plans;return plans
    def run_piece(self,module,x,outputs,plan):
        segments,starts,widths,strides=plan
        targets=(ctypes.c_void_p*len(segments))(*[outputs[s[3]].data_ptr()+s[4]*2 for s in segments])
        bias=module.bias;ptr=bias.data_ptr() if bias is not None else None
        t=time.perf_counter()
        self.status(self.lib.mm_run_prepared_scatter_bf16(self.ensure(module),x.numel()//x.shape[-1],len(segments),starts,widths,strides,targets,ptr))
        self.calls+=1
        if self.profile:self.events.append({'core':self.core[id(module)],'n':module.out_features,'start':t,'end':time.perf_counter()})
    def group(self,modules,x):
        entry=self.pack(modules)
        if entry is None:return super().group(modules,x)
        packed,_=entry;plans=self.plan(modules,packed)
        for m in packed:self.ensure(m)
        inp=x.contiguous();rows=x.numel()//x.shape[-1]
        self.status(self.lib.mm_prepare_bf16(self.ensure(packed[0]),rows,inp.data_ptr()))
        self.status(self.lib.mm_copy_prepared(self.ensure(packed[1]),self.ensure(packed[0]),rows))
        outputs={id(m):self.output_buffer(m,x) for m in modules}
        jobs=[self.pool.submit(self.run_piece,m,x,outputs,plan) for m,plan in zip(packed,plans)]
        error=None;range_failed=False
        for job in jobs:
            try:job.result()
            except OutputRangeError:range_failed=True
            except Exception as exc:
                if error is None:error=exc
        if error is not None:raise error
        self.stats['packed_groups']+=1;self.stats['shared_input_prepares']+=1;self.stats['parallel_jobs']+=2
        if range_failed:
            self.stats['packed_range_fallbacks']+=1;self.cpu_fallbacks+=len(modules)
            return {id(m):m._bp_original(x) for m in modules}
        return outputs
