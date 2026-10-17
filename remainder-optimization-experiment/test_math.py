"""Focused SIMD numeric boundaries, dtype ownership, same BF16 boundaries."""
import ctypes,json
from pathlib import Path
import torch
from cpu_math import lib,activation

def run():
    old=ctypes.CDLL('/release/libd3_native_norm.so')
    old.d3_norm.argtypes=lib.d3_norm.argtypes
    torch.manual_seed(469)
    rows=[]
    def difference(actual,reference):
        assert torch.equal(actual.isnan(),reference.isnan())
        assert torch.equal(actual.isinf(),reference.isinf())
        assert torch.equal(actual[actual.isinf()].signbit(),reference[reference.isinf()].signbit())
        finite=torch.isfinite(actual)&torch.isfinite(reference)
        both_zero=(actual==0)&(reference==0)
        active=finite&~both_zero
        a,b=actual.view(torch.int16).int(),reference.view(torch.int16).int()
        ulp=(a[active]-b[active]).abs().max().item() if active.any() else 0
        assert ulp<=2,ulp
        delta=(actual[finite].float()-reference[finite].float()).abs()
        return {'bitwise_equal':torch.equal(a,b),'max_bf16_ulp':ulp,'max_finite_abs':delta.max().item() if delta.numel() else 0}
    with torch.inference_mode():
        values=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
        up=torch.randn_like(values)
        for kind in (0,1,2):
            reference=torch.sigmoid(values) if kind==2 else torch.nn.functional.silu(values)
            if kind==1:reference=reference*up
            actual=activation(values,up if kind==1 else None,kind)
            rows.append({'test':'all_bf16_patterns','kind':kind,**difference(actual,reference)})
        for count in (0,1,2,3,5,7,17,1023,1024,1025,4097):
            x=torch.randn(count*2,dtype=torch.bfloat16)[::2];before=x.clone();up=torch.randn_like(x)
            for kind in (0,1,2):
                ref=torch.sigmoid(x) if kind==2 else torch.nn.functional.silu(x)
                if kind==1:ref=ref*up
                actual=activation(x,up if kind==1 else None,kind)
                d=difference(actual,ref) if count else {'empty':True}
                assert torch.equal(x,before)
                rows.append({'test':'tail_and_noncontiguous','count':count,'kind':kind,**d})
        for count,dim in ((1,7),(17,127),(115,128),(129,1024),(115,3584)):
            x=torch.randn(count,dim,dtype=torch.bfloat16);w=torch.randn(dim,dtype=torch.bfloat16)*.2
            gate=torch.randn_like(x)*4
            for kind in (0,1):
                actual=torch.empty_like(x);reference=torch.empty_like(x)
                args=(x.data_ptr(),w.data_ptr(),gate.data_ptr() if kind else None)
                old.d3_norm(*args,reference.data_ptr(),count,dim,1e-6,kind)
                lib.d3_norm(*args,actual.data_ptr(),count,dim,1e-6,kind)
                d=difference(actual,reference)
                if not kind:assert d['bitwise_equal']
                rows.append({'test':'same_reduction_norm','rows':count,'dim':dim,'kind':kind,**d})
    result={'passed':True,'rows':rows,'exhaustive_patterns':65536,'scope':'All BF16 bit patterns for SiLU/sigmoid and SiLU-BF16-multiply-BF16, tails/noncontiguous/empty, norm vs unchanged original own C kernel; <=2 BF16 ULP per element, exact ordinary norm.'}
    Path('/work/logs/math-unit-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print('MATH_UNITS',len(rows),'passed',flush=True)
    return result
