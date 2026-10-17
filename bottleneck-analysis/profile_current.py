"""Isolated attribution of the exact deployed three-thread arithmetic path."""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time
import types
import torch

p=argparse.ArgumentParser()
p.add_argument('--model',required=True)
p.add_argument('--count',type=int,default=12)
a=p.parse_args()
sys.path[:0]=['/release','/base',a.model]
from d3_runtime import D3
from cpu_optimized_server import optimize
from rknn_linear import Engine,install,uninstall
from cpu_attention_optimized import install_conv,install_scan,install_sdpa
from native_norm import install_norm,install_native_conv
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

torch.set_num_threads(3)
model=D3.from_pretrained(a.model,device='cpu',verify='full')
assert model.attention_mode=='noncausal_full_attention' and len(model.encoder.stack)==12
optimize(model)
engine=Engine();selected=install(model,engine)
assert len(selected)==78
native={}
restore=[install_conv(),install_scan(128),install_sdpa(),install_norm(model.backbone,native),install_native_conv(native)]
body=json.loads(Path('/base/request.json').read_text())
out=Path('/work/logs')
result={'settings':{'torch_threads':torch.get_num_threads(),'affinity':sorted(os.sched_getaffinity(0)),
                   'env':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OMP_WAIT_POLICY','GOMP_SPINCOUNT')},
                   'torch':torch.__version__},'source_sha256':{name:hashlib.sha256(Path('/release',name).read_bytes()).hexdigest() for name in ('cpu_attention_optimized.py','native_norm.py','native_norm.c')},
        'scope':'115 tokens, single question, concurrency 1; fresh isolated process with exact deployed kernels; no service modification.'}

def invoke():return model.system_one(state=body['state'],questions=body['questions'])
def save(): (out/'profile-current.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
def numeric(x,path=''):
    found={}
    if isinstance(x,dict):
        for k,v in x.items():
            if k not in ('confidence','legend'):found.update(numeric(v,path+'/'+k))
    elif isinstance(x,(int,float)):found[path]=x
    return found

def measure(count):
    values=[]
    for i in range(count):
        t=time.perf_counter();response=invoke();elapsed=time.perf_counter()-t
        assert numeric(response['answers'])==numeric(reference['answers'])
        values.append(elapsed)
        print('RAW',i,elapsed,flush=True)
    return {'seconds':values,'median':statistics.median(values),'min':min(values),'max':max(values),'response':response}

class Attribution:
    def __init__(self):self.stack=[];self.rows=defaultdict(lambda:{'inclusive':0.,'exclusive':0.,'calls':0})
    def enter(self,key):self.stack.append([key,time.perf_counter(),0.])
    def leave(self):
        key,start,children=self.stack.pop();elapsed=time.perf_counter()-start
        row=self.rows[key];row['inclusive']+=elapsed;row['exclusive']+=elapsed-children;row['calls']+=1
        if self.stack:self.stack[-1][2]+=elapsed
    @contextmanager
    def span(self,key):
        self.enter(key)
        try:yield
        finally:self.leave()

wrapped=[];hooks=[];shapes=defaultdict(lambda:{'calls':0,'flops':0,'weight_bytes_fp16':0})
prof=Attribution()
def wrap(owner,name,key):
    original=getattr(owner,name)
    def call(*args,**kwargs):
        with prof.span(key):return original(*args,**kwargs)
    setattr(owner,name,call);wrapped.append((owner,name,original))

try:
    reference=invoke()
    assert native['norm_calls']>0 and native['conv_calls']>0
    result['uninstrumented_before']=measure(a.count);save()
    # The C shim remains the same algorithm, layout, memcpy calls and SDK calls.
    # Only own bridge source is instrumented; no vendor binary is inspected.
    timing_lib=ctypes.CDLL('/work/libmatmul_profile.so')
    timing_lib.mm_error.restype=ctypes.c_char_p
    timing_lib.mm_create.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];timing_lib.mm_create.restype=ctypes.c_void_p
    timing_lib.mm_run.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];timing_lib.mm_run.restype=ctypes.c_int
    timing_lib.mm_destroy.argtypes=[ctypes.c_void_p]
    timing_lib.mm_profile_read.argtypes=[ctypes.POINTER(ctypes.c_double)]
    # Swap engine after restoring forwards and releasing every original handle.
    uninstall(model);engine.close();engine=Engine();engine.lib=timing_lib;install(model,engine)
    switched=invoke();assert switched==reference
    result['clock_shim_uninstrumented']=measure(5);save()
    for name in ('prepare','run','respond'):wrap(model,name,'request.'+name)
    for owner,name,key in [(qwen,'torch_chunk_gated_delta_rule','scan'),(qwen,'causal_conv1d_fn','causal_conv'),
         (qwen,'apply_rotary_pos_emb','rotary'),(torch.nn.functional,'scaled_dot_product_attention','sdpa'),
         (torch.linalg,'solve_triangular','triangular_solve')]:wrap(owner,name,key)
    for name,module in model.backbone.named_modules():
        kind=type(module).__name__
        if isinstance(module,torch.nn.Linear):
            category='linear.rknn' if hasattr(module,'_rknn_cpu_forward') else 'linear.cpu'
        elif 'RMSNorm' in kind:category='rmsnorm'
        elif 'GatedDeltaNet' in kind:category='delta.elementwise_and_dispatch'
        elif kind=='Qwen3_5Attention':category='attention.elementwise_and_dispatch'
        elif isinstance(module,torch.nn.Embedding):category='embedding'
        elif isinstance(module,torch.nn.SiLU):category='silu'
        elif 'MLP' in kind:category='mlp.elementwise_and_dispatch'
        elif 'DecoderLayer' in kind:category='block.residual_and_dispatch'
        else:category='module.other.'+kind
        def before(module,inputs,key=category):
            prof.enter(key)
            if isinstance(module,torch.nn.Linear) and inputs and isinstance(inputs[0],torch.Tensor):
                rows=inputs[0].numel()//inputs[0].shape[-1]
                k,n=module.in_features,module.out_features
                tag=f'{key}:M{rows}:K{k}:N{n}'
                item=shapes[tag];item['calls']+=1;item['flops']+=2*rows*k*n;item['weight_bytes_fp16']=k*n*2
        def after(module,inputs,output):prof.leave()
        hooks.extend([module.register_forward_pre_hook(before),module.register_forward_hook(after)])
    # Same Engine.run operations; spans only separate existing stages.
    original_run=engine.run
    def measured_run(self,handle,inputs,out_features):
        start=time.perf_counter()
        with prof.span('adapter.convert_fp16'):
            flat=inputs.reshape(-1,inputs.shape[-1]).to(torch.float16).contiguous()
        with prof.span('adapter.finite_check'):
            if not torch.isfinite(flat).all():raise ValueError('FP16 activation overflow')
        with prof.span('adapter.allocate_fp32'):
            output=torch.empty((flat.shape[0],out_features),dtype=torch.float32)
        with prof.span('adapter.bridge_ffi'):
            if self.lib.mm_run(handle,flat.shape[0],flat.data_ptr(),output.data_ptr()):raise self.error()
        self.calls+=1;self.seconds+=time.perf_counter()-start
        return output.reshape(*inputs.shape[:-1],out_features)
    engine.run=types.MethodType(measured_run,engine)
    records=[]
    for i in range(8):
        prof=Attribution();shapes.clear();timing_lib.mm_profile_reset()
        before=resource.getrusage(resource.RUSAGE_SELF)
        with prof.span('request.other'):
            response=invoke()
        after=resource.getrusage(resource.RUSAGE_SELF)
        assert response==reference
        native_times=(ctypes.c_double*6)();timing_lib.mm_profile_read(native_times)
        total=prof.rows['request.other']['inclusive']
        assert abs(sum(v['exclusive'] for v in prof.rows.values())-total)<1e-8
        records.append({'seconds':total,'timings':dict(prof.rows),'native_bridge':dict(zip(['calls','shape_bind','input_copy','input_padding','rknn_matmul_run','output_copy'],native_times)),
             'linear_shapes':dict(shapes),'cpu_user_seconds':after.ru_utime-before.ru_utime,'cpu_system_seconds':after.ru_stime-before.ru_stime,
             'minor_faults':after.ru_minflt-before.ru_minflt,'voluntary_switches':after.ru_nvcsw-before.ru_nvcsw,'involuntary_switches':after.ru_nivcsw-before.ru_nivcsw})
        print('PROFILE',i,total,json.dumps(records[-1]['native_bridge']),flush=True)
    result['instrumented']=records;result['instrumented_outputs_exactly_equal']=True;save()
    for hook in hooks:hook.remove()
    hooks.clear()
    for owner,name,original in reversed(wrapped):setattr(owner,name,original)
    wrapped.clear();engine.run=original_run
    # original_run remains bound to this engine and the timing bridge.
    result['uninstrumented_after']=measure(a.count)
    result['native_stats']=native
    result['model_parameters']=sum(p.numel() for p in model.backbone.parameters())
    result['passed']=True;save()
finally:
    for hook in hooks:hook.remove()
    for owner,name,original in reversed(wrapped):setattr(owner,name,original)
    for callback in reversed(restore):callback()
    uninstall(model);engine.close()
