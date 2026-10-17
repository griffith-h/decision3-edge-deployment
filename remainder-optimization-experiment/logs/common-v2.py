"""One model/engine per process, frozen candidate, scoped restoration."""
import ctypes,hashlib,json,math,os,statistics,sys,time
from pathlib import Path
import torch

def paths(model_path):sys.path[:0]=['/release','/base','/combined','/scan','/work',model_path]
def load(model_path,profile=False,control=None,half_output=False):
    paths(model_path)
    from d3_runtime import D3
    from cpu_optimized_server import optimize
    from cpu_attention_optimized import install_conv,install_scan,install_sdpa
    from native_norm import install_norm,install_native_conv
    from adapter import Engine,install
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    from scan import make_scan
    torch.set_num_threads(3)
    model=D3.from_pretrained(model_path,device='cpu',verify='full');model.model_name='vllm-sr/d3-edge'
    assert len(model.encoder.stack)==12 and model.attention_mode=='noncausal_full_attention'
    optimize(model);native={}
    restores=[install_conv(),install_scan(128),install_sdpa(),install_norm(model.backbone,native),install_native_conv(native)]
    qwen.torch_chunk_gated_delta_rule=make_scan(qwen.torch_chunk_gated_delta_rule,tile=16,fused=5)
    if control is not None:
        from cpu_math import install as install_math
        restores.append(install_math(model,control))
    engine=Engine(fusion=True,parallel=True)
    if profile or half_output:
        old=engine.lib;lib=ctypes.CDLL('/work/libhalf_profile.so' if profile and half_output else '/work/libhalf_bridge.so' if half_output else '/work/libfused_profile.so')
        for name in ('mm_error','mm_create_core','mm_run','mm_destroy','mm_prepare_bf16','mm_copy_prepared','mm_run_prepared_bf16','mm_run_bf16'):
            src=getattr(old,name);dst=getattr(lib,name);dst.argtypes=src.argtypes;dst.restype=src.restype
        if profile:lib.mm_profile_read.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_int]
        engine.lib=lib
    restore_engine=install(model,engine)
    engine.restore_forwards=restore_engine
    def close():
        if engine.restore_forwards is not None:engine.restore_forwards();engine.restore_forwards=None
        engine.close()
        for restore in reversed(restores):restore()
    return model,engine,native,close
def numeric(value,path=''):
    out={}
    if isinstance(value,dict):
        for k,v in value.items():
            if k not in ('confidence','legend'):out.update(numeric(v,path+'/'+k))
    elif isinstance(value,(int,float)):out[path]=value
    return out
def compare(actual,expected):
    x,y=numeric(actual['answers']),numeric(expected['answers']);assert x.keys()==y.keys()
    assert all(math.isfinite(v) for v in x.values())
    delta=max(abs(x[k]-y[k]) for k in x)
    same=all(v.get('choice')==expected['answers'][k].get('choice') for k,v in actual['answers'].items())
    return {'max_delta':delta,'choices_unchanged':same,'passed':delta<=.01 and same}
def stats(values):
    return {'seconds':values,'count':len(values),'median':statistics.median(values),'min':min(values),'max':max(values),
            'p95_nearest_rank':sorted(values)[math.ceil(.95*len(values))-1]}
def sources():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ('/work','/scan','/combined','/release') for p in Path(folder).iterdir() if p.suffix in ('.py','.c','.cpp','.so')}
def state():
    return {'torch_threads':torch.get_num_threads(),'affinity':sorted(os.sched_getaffinity(0)),
            'cpu_khz':Path('/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq').read_text().strip(),
            'npu_hz':Path('/sys/class/devfreq/27700000.npu/cur_freq').read_text().strip()}
