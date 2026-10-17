"""Isolated fusion/parallel ablations; per-parent-call caches, original CPU fallback."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
from pathlib import Path
import time
from types import MethodType
import torch
from rknn_linear import Engine as OriginalEngine

class Engine(OriginalEngine):
    def __init__(self,fusion=False,parallel=False,split_core=True):
        super().__init__()
        self.lib=ctypes.CDLL(str(Path(__file__).with_name('libfused_bridge.so')))
        self.lib.mm_error.restype=ctypes.c_char_p
        self.lib.mm_create_core.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
        self.lib.mm_create_core.restype=ctypes.c_void_p
        self.lib.mm_run.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];self.lib.mm_run.restype=ctypes.c_int
        self.lib.mm_destroy.argtypes=[ctypes.c_void_p]
        for name,count in [('mm_prepare_bf16',3),('mm_copy_prepared',3),('mm_run_prepared_bf16',4),('mm_run_bf16',5)]:
            fn=getattr(self.lib,name)
            fn.argtypes=([ctypes.c_void_p,ctypes.c_int]+[ctypes.c_void_p]*(count-2)) if name!='mm_copy_prepared' else [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
            fn.restype=ctypes.c_int
        self.fusion,self.parallel,self.split_core=fusion,parallel,split_core
        self.pool=ThreadPoolExecutor(max_workers=2) if parallel else None
        self.core={};self.bucket={};self.groups=[];self.stats={'groups':0,'shared_input_prepares':0,'parallel_jobs':0,'cpu_fallbacks':0}

    def create(self,weight,core=0):
        start=time.perf_counter();half=weight.detach().to(torch.float16).t().contiguous()
        if not torch.isfinite(half).all():raise ValueError('FP16 weight overflow')
        handle=self.lib.mm_create_core(weight.shape[1],weight.shape[0],half.data_ptr(),core)
        if not handle:raise self.error()
        self.handles.append(handle);self.create_seconds+=time.perf_counter()-start
        return handle

    def ensure(self,module):
        if module._bp_handle is None:module._bp_handle=self.create(module.weight,self.core.get(id(module),0))
        return module._bp_handle

    def status(self,status):
        if status==-2:raise ValueError('FP16 activation overflow')
        if status:raise self.error()

    def eligible(self,module,x):
        return x.device.type=='cpu' and x.dtype==torch.bfloat16 and 0<x.numel()//x.shape[-1]<=512

    def output_buffer(self,module,x):return torch.empty((*x.shape[:-1],module.out_features),dtype=torch.bfloat16)
    def bias(self,module):return module.bias.detach().float().contiguous() if module.bias is not None else None

    def execute(self,module,x,prepared=False,output=None):
        handle=self.ensure(module)
        if not self.fusion:
            output=OriginalEngine.run(self,handle,x,module.out_features)
            if module.bias is not None:output+=module.bias.float()
            return output.to(x.dtype)
        start=time.perf_counter();rows=x.numel()//x.shape[-1]
        output=self.output_buffer(module,x) if output is None else output
        bias=self.bias(module);bp=bias.data_ptr() if bias is not None else None
        if prepared:self.status(self.lib.mm_run_prepared_bf16(handle,rows,bp,output.data_ptr()))
        else:
            contiguous=x.contiguous()
            self.status(self.lib.mm_run_bf16(handle,rows,contiguous.data_ptr(),bp,output.data_ptr()))
        self.calls+=1;self.seconds+=time.perf_counter()-start
        return output

    def group(self,modules,x):
        # Handles are created on the model owner thread, before worker submission.
        for module in modules:self.ensure(module)
        outputs={}
        if self.fusion:
            rows=x.numel()//x.shape[-1];contiguous=x.contiguous()
            first=self.ensure(modules[0])
            self.status(self.lib.mm_prepare_bf16(first,rows,contiguous.data_ptr()))
            for module in modules[1:]:self.status(self.lib.mm_copy_prepared(self.ensure(module),first,rows))
            outputs={id(m):self.output_buffer(m,x) for m in modules}
            self.stats['shared_input_prepares']+=1
        # One task per core bucket, so a three-projection fork never queues two
        # concurrent jobs onto the same handle or races its mutable input buffer.
        buckets=[[m for m in modules if self.bucket[id(m)]==i] for i in range(2)]
        def batch(items):return [(id(m),self.execute(m,x,prepared=self.fusion,output=outputs.get(id(m)))) for m in items]
        futures=[self.pool.submit(batch,bucket) for bucket in buckets if bucket]
        self.stats['groups']+=1;self.stats['parallel_jobs']+=len(futures)
        error=None;results={}
        # Join every job before propagating an error or allowing buffer reuse.
        for future in futures:
            try:results.update(future.result())
            except Exception as exc:
                if error is None:error=exc
        if error is not None:raise error
        return results

    def close(self):
        if self.pool is not None:self.pool.shutdown(wait=True);self.pool=None
        super().close()

def install(model,engine):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    selected=[];parents=[]
    for name,module in model.backbone.named_modules():
        if 'language_model.layers.' not in name or not isinstance(module,torch.nn.Linear):continue
        k,n=module.in_features,module.out_features
        if k>10240 or k%16 or n%8 or k*n<262144:continue
        module._bp_original=module.forward;module._bp_handle=None;selected.append(module)
    selected_ids={id(m) for m in selected}
    scopes={}
    if engine.parallel:
        for name,parent in model.backbone.named_modules():
            if isinstance(parent,qwen.Qwen3_5MLP):names=['gate_proj','up_proj']
            elif isinstance(parent,qwen.Qwen3_5Attention):names=['q_proj','k_proj','v_proj']
            elif isinstance(parent,qwen.Qwen3_5GatedDeltaNet):names=['in_proj_qkv','in_proj_z']
            else:continue
            members=[getattr(parent,key) for key in names]
            if not all(id(m) in selected_ids for m in members):continue
            scope={'active':False,'used':False,'input':None,'outputs':{},'members':members}
            for index,module in enumerate(members):
                scopes[id(module)]=scope;bucket=(0 if index==0 else 1)
                engine.bucket[id(module)]=bucket
                engine.core[id(module)]=1+(bucket if engine.split_core else 0)
            original=parent.forward
            def parent_forward(self,*args,_scope=scope,_original=original,**kwargs):
                if _scope['active']:return _original(*args,**kwargs)
                _scope.update(active=True,used=False,input=None,outputs={})
                try:return _original(*args,**kwargs)
                finally:_scope.update(active=False,used=False,input=None,outputs={})
            parent.forward=MethodType(parent_forward,parent);parents.append((parent,original))
            engine.groups.append({'parent':name,'members':names})
    for module in selected:
        def forward(self,x):
            if not engine.eligible(self,x):
                engine.stats['cpu_fallbacks']+=1;engine.cpu_fallbacks+=1
                return self._bp_original(x)
            scope=scopes.get(id(self))
            if scope is not None and scope['active']:
                if not scope['used']:
                    scope['used']=True;scope['input']=x
                    scope['outputs']=engine.group(scope['members'],x)
                if scope['input'] is x and id(self) in scope['outputs']:
                    return scope['outputs'].pop(id(self))
            return engine.execute(self,x)
        module.forward=MethodType(forward,module)
    def restore():
        for parent,original in parents:parent.forward=original
        for module in selected:
            module.forward=module._bp_original
            del module._bp_original;del module._bp_handle
    return restore
