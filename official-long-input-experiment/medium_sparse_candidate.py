"""Development-only medium-input BF16 rounding repair using original CPU GEMM.

Read completed FP32 contexts after workers join. All handles stay in the
existing Owner banks; native MLP may create additional single-down contexts.
Model weights stay unchanged. Original115/615 and T>2048 paths bypass this adapter.
Full-component and full-request qualification are still required.
"""
import ctypes,hashlib,json
from pathlib import Path
import torch
from sparse_rounding import SparseRepair

class MediumSparse:
    def __init__(self,owner):
        self.owner=owner;self.engine=owner.engine;self.native=owner.native_mlp
        self.closed=False;self.group_active=False;self.repair=SparseRepair()
        self.initial_native_handles=len(self.native.handles)
        self.counts={'projection_groups':0,'projection_singles':0,'mlp_chunks':0,'read_errors':0}
        self.original_group=self.engine.group;self.original_execute=self.engine.execute;self.original_mlp=self.native.run
        self.activations={id(state[2][0]):state[-1] for state in owner.long_mlp.states}
        root=Path(__file__).parent
        def library(manifest_name,libname):
            manifest=json.loads((root/manifest_name).read_text())
            for name,digest in manifest['sha256'].items():
                if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Precision reader source/library mismatch')
            return ctypes.CDLL(str(root/libname))
        self.norm=library('projection-read-build.json','libprojection_read_probe.so')
        self.norm.d3_projection_read_fp32.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        self.norm.d3_projection_read_fp32.restype=ctypes.c_int
        self.raw_native=library('linear-precision-build.json','liblinear_precision_probe.so')
        self.raw_native.d3_probe_copy_fp32.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
        self.raw_native.d3_probe_copy_fp32.restype=ctypes.c_int
        def execute(module,x,prepared=False,output=None):
            before=self.engine.cpu_fallbacks;result=self.original_execute(module,x,prepared,output)
            if self.qualified() and not self.group_active and self.engine.cpu_fallbacks==before:
                result=self.correct_norm(module,x);self.counts['projection_singles']+=1
            return result
        def group(modules,x):
            if not self.qualified():return self.original_group(modules,x)
            if self.group_active:raise RuntimeError('Unexpected nested projection correction')
            before=self.engine.cpu_fallbacks;self.group_active=True
            try:result=self.original_group(modules,x)
            finally:self.group_active=False
            if self.engine.cpu_fallbacks!=before:return result
            entry=self.engine.pack(modules)
            if entry is None:result={id(m):self.correct_norm(m,x) for m in modules}
            else:
                packed,sizes=entry
                joined=torch.cat([self.correct_norm(m,x,allow_inference_weight=True) for m in packed],dim=-1)
                result={id(m):part.contiguous() for m,part in zip(modules,joined.split(sizes,dim=-1))}
            self.counts['projection_groups']+=1;return result
        def mlp(members,x):
            if not self.qualified() or not owner.long_mlp.active:return self.original_mlp(members,x)
            gate,up,down=members
            # Native.group joins both workers before either raw C is read.
            self.native.group((gate,up),x)
            gv=self.correct_native(gate,x);uv=self.correct_native(up,x)
            product=self.activations[id(gate)](gv)*uv
            self.native.execute(down,product)
            value=self.correct_native(down,product)
            self.native.calls+=1;self.counts['mlp_chunks']+=1
            return value
        self.execute_wrapper=execute;self.group_wrapper=group;self.mlp_wrapper=mlp
        self.engine.execute=execute;self.engine.group=group;self.native.run=mlp
    def qualified(self):
        return (not self.closed and getattr(getattr(self.owner,'medium_precision',None),'active',False)
                and getattr(getattr(self.owner,'long_projection',None),'active',False))
    def correct_norm(self,module,x,allow_inference_weight=False):
        rows=x.numel()//x.shape[-1];raw=torch.empty((rows,module.out_features),dtype=torch.float32)
        rc=self.norm.d3_projection_read_fp32(self.engine.ensure(module),rows,module.in_features,module.out_features,raw.data_ptr())
        if rc:self.counts['read_errors']+=1;raise ValueError('Normal FP32 context read failed: '+str(rc))
        result=self.repair.repair(module,x.reshape(rows,x.shape[-1]),raw,allow_inference_weight=allow_inference_weight)
        return result.reshape(*x.shape[:-1],module.out_features)
    def correct_native(self,module,x,handle=None):
        rows=x.numel()//x.shape[-1];raw=torch.empty((rows,module.out_features),dtype=torch.float32)
        rc=self.raw_native.d3_probe_copy_fp32(self.native.ensure(module) if handle is None else handle,rows,raw.data_ptr())
        if rc:self.counts['read_errors']+=1;raise ValueError('Native FP32 context read failed: '+str(rc))
        result=self.repair.repair(module,x.reshape(rows,x.shape[-1]),raw)
        return result.reshape(*x.shape[:-1],module.out_features)
    def snapshot(self):return {**self.counts,'repair':dict(self.repair.counts),
      'initial_native_handles':self.initial_native_handles,'current_native_handles':len(self.native.handles),
      'base_down_handles':sum(id(state[2][2]) in self.native.handles for state in self.owner.long_mlp.states)}
    def close(self):
        if self.closed:return
        if self.group_active:raise RuntimeError('Cannot restore precision adapter while workers are active')
        self.closed=True
        if self.engine.execute is self.execute_wrapper:self.engine.execute=self.original_execute
        if self.engine.group is self.group_wrapper:self.engine.group=self.original_group
        if self.native.run is self.mlp_wrapper:self.native.run=self.original_mlp
        self.repair.cache.clear()

def install(owner):
    owner.medium_sparse=MediumSparse(owner)
    return owner.medium_sparse.close
