"""Small focused test of new FP16 output range, CPU fallback and recovery."""
import sys,ctypes,json
from pathlib import Path
sys.path[:0]=['/base','/combined','/work']
import torch
from adapter import Engine
from safe_engine import SafeEngine
torch.set_num_threads(3)
def configure(engine,path):
    old=engine.lib;lib=ctypes.CDLL(path)
    for name in ('mm_error','mm_create_core','mm_run','mm_destroy','mm_prepare_bf16','mm_copy_prepared','mm_run_prepared_bf16','mm_run_bf16'):
        src=getattr(old,name);dst=getattr(lib,name);dst.argtypes=src.argtypes;dst.restype=src.restype
    engine.lib=lib
def module(fill=1):
    m=torch.nn.Linear(1024,32,bias=False,dtype=torch.bfloat16)
    with torch.no_grad():m.weight.fill_(fill)
    m._bp_handle=None
    m._bp_original=lambda x:torch.nn.functional.linear(x.float(),m.weight.float()).bfloat16()
    return m
unsafe=Engine(fusion=True,parallel=False);configure(unsafe,'/work/libhalf_bridge.so')
safe=SafeEngine(fusion=True,parallel=True);configure(safe,'/work/libsafe_half_bridge.so')
u=module();m=module();zero=module(0);records=[]
try:
    with torch.inference_mode():
        for count in (1,115,128,129,512,1):
            x=torch.full((count,1024),.125,dtype=torch.bfloat16)
            out=safe.execute(m,x);assert torch.equal(out,m._bp_original(x))
            records.append({'rows':count,'exact_valid_output':True})
        x=torch.full((1,1024),1024,dtype=torch.bfloat16)
        unsafe_out=unsafe.execute(u,x)
        ref=m._bp_original(x);assert torch.isfinite(ref).all() and ref[0,0].item()==1048576
        raw_difference={'unsafe_first_value':str(unsafe_out[0,0].item()),'cpu_first_value':ref[0,0].item(),
                        'unsafe_differs':not torch.equal(unsafe_out,ref)}
        assert raw_difference['unsafe_differs']
        out=safe.execute(m,x);assert torch.equal(out,ref)
        assert safe.stats['half_output_cpu_fallbacks']==1
        # A two-projection group with one overflowing member and one finite
        # member must return both valid results, then recover on the next call.
        for i,item in enumerate((m,zero)):
            safe.core[id(item)]=i+1;safe.bucket[id(item)]=i
        outputs=safe.group([m,zero],x)
        assert torch.equal(outputs[id(m)],ref) and not outputs[id(zero)].any()
        assert safe.stats['half_output_cpu_fallbacks']==2
        normal=torch.full((1,1024),.125,dtype=torch.bfloat16)
        outputs=safe.group([m,zero],normal)
        assert torch.equal(outputs[id(m)],m._bp_original(normal)) and not outputs[id(zero)].any()
        assert safe.stats['half_output_cpu_fallbacks']==2
        invalid=normal.clone();invalid[0,0]=float('nan')
        try:safe.execute(m,invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid input validation must remain unchanged')
        assert torch.equal(safe.execute(m,normal),m._bp_original(normal))
    result={'passed':True,'valid_shapes':records,'range_boundary':raw_difference,
            'fallbacks':safe.stats['half_output_cpu_fallbacks'],'mixed_group_and_recovery_passed':True,'input_validation_preserved':True,
            'scope':'Finite BF16 inputs and FP16 weights can produce out-of-range FP16 output. New candidate detects saturation/nonfinite during output conversion and delegates that projection to the original CPU forward.'}
    Path('/work/logs/output-range-unit.json').write_text(json.dumps(result,indent=2)+'\n')
    print('OUTPUT_RANGE_UNITS',result,flush=True)
finally:unsafe.close();safe.close()
