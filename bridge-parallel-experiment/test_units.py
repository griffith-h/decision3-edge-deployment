"""Focused native-conversion, shape, fallback and worker-lifecycle checks."""
import ctypes,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys,time,types
import torch
sys.path[:0]=['/base','/work']
from adapter import Engine,install
from cpu_optimized_server import optimize

def run():
    torch.set_num_threads(3);torch.manual_seed(3471)
    engine=Engine(fusion=True)
    lib=engine.lib
    lib.mm_convert_input.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];lib.mm_convert_input.restype=ctypes.c_int
    lib.mm_convert_output.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    count=0
    try:
        bits=torch.arange(65536,dtype=torch.int32).to(torch.uint16)
        values=bits.view(torch.bfloat16);half=torch.empty(65536,dtype=torch.float16)
        status=lib.mm_convert_input(values.data_ptr(),half.data_ptr(),values.numel())
        reference=values.to(torch.float16)
        finite_bf=torch.isfinite(values)
        assert status==-2
        assert torch.equal(half.view(torch.uint16)[finite_bf],reference.view(torch.uint16)[finite_bf])
        assert torch.equal(torch.isfinite(half),torch.isfinite(reference))
        count+=65536
        for n in (1,2,3,4,5,7,17):
            value=torch.randn(n,dtype=torch.bfloat16);converted=torch.empty(n,dtype=torch.float16)
            assert lib.mm_convert_input(value.data_ptr(),converted.data_ptr(),n)==0
            assert torch.equal(converted,value.to(torch.float16));count+=1
        raw=torch.randint(0,2**32,(100003,),dtype=torch.int64).to(torch.uint32)
        f=raw.view(torch.float32).contiguous();b=torch.empty(f.shape,dtype=torch.bfloat16)
        lib.mm_convert_output(f.data_ptr(),b.data_ptr(),f.numel())
        assert torch.equal(b.view(torch.uint16),f.to(torch.bfloat16).view(torch.uint16))
        count+=f.numel()
        module=torch.nn.Linear(32,32,bias=True).to(torch.bfloat16)
        module._bp_handle=engine.create(module.weight)
        for rows in (1,115,127,128,129,256,257,512,128,1):
            x=torch.randn(rows,64,dtype=torch.bfloat16)[:,::2]
            engine.fusion=False;legacy=engine.execute(module,x)
            engine.fusion=True;fused=engine.execute(module,x)
            assert torch.equal(fused,legacy),(rows,(fused.float()-legacy.float()).abs().max().item())
            count+=1
        for value in (float('nan'),float('inf'),-float('inf'),65536.,-65536.):
            x=torch.zeros(3,32,dtype=torch.bfloat16);x[1,1]=value
            try:engine.execute(module,x)
            except ValueError as exc:assert 'overflow' in str(exc)
            else:raise AssertionError('Invalid/overflow input accepted')
            # Recovery must fully overwrite previously failed native input.
            x.zero_();engine.fusion=False;ref=engine.execute(module,x)
            engine.fusion=True;assert torch.equal(ref,engine.execute(module,x));count+=1
        invalid=torch.zeros(513,32,dtype=torch.bfloat16)
        assert lib.mm_prepare_bf16(module._bp_handle,513,invalid.data_ptr())!=0
        count+=1
        # Test the adapter's actual saved CPU fallback, not a duplicated formula.
        backbone=torch.nn.Module();backbone.language_model=torch.nn.Module()
        backbone.language_model.layers=torch.nn.ModuleList([torch.nn.Module()])
        linear=torch.nn.Linear(1024,512,bias=True).to(torch.bfloat16)
        backbone.language_model.layers[0].linear=linear
        toy=types.SimpleNamespace(backbone=backbone);optimize(toy)
        saved=linear.forward
        restore=install(toy,engine)
        try:
            for x in (torch.randn(513,1024,dtype=torch.bfloat16),torch.randn(3,1024,dtype=torch.float32),torch.empty(0,1024,dtype=torch.bfloat16)):
                expected=saved(x);actual=linear(x)
                assert torch.equal(expected,actual);count+=1
        finally:restore()
        # Error propagation waits for every worker before releasing/reusing buffers.
        fake=Engine.__new__(Engine);fake.fusion=False;fake.pool=ThreadPoolExecutor(max_workers=2)
        fake.stats={'groups':0,'parallel_jobs':0};members=[object(),object()]
        fake.bucket={id(m):i for i,m in enumerate(members)};fake.ensure=lambda m:id(m)
        completed=[]
        def execute(m,x,**kwargs):
            if m is members[0]:raise ValueError('injected worker failure')
            time.sleep(.03);completed.append(True);return torch.zeros(1)
        fake.execute=execute
        try:
            try:fake.group(members,torch.zeros(1))
            except ValueError as exc:assert str(exc)=='injected worker failure'
            else:raise AssertionError('Worker failure swallowed')
            assert completed==[True];count+=1
        finally:fake.pool.shutdown(wait=True)
        result={'passed':True,'converter_values_and_focused_cases':count,'exhaustive_bf16_input_patterns':65536,
                'random_fp32_output_patterns':100003,'dynamic_rows':[1,115,127,128,129,256,257,512,128,1],
                'noncontiguous_and_biased_outputs_bitwise_equal':True,'nonfinite_and_fp16_overflow_rejected':True,
                'failed_input_recovery_passed':True,'cpu_fallback_exact':True,'worker_exception_join_passed':True}
        Path('/work/logs/unit-results.json').write_text(json.dumps(result,indent=2)+'\n')
        print('FOCUSED_UNITS_PASSED',json.dumps(result),flush=True)
    finally:engine.close()

if __name__=='__main__':run()
