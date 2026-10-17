import ctypes,json,torch,sys
from pathlib import Path
lib=ctypes.CDLL('/work/libconversion_bridge.so')
for name in ('mm_convert_input','mm_fast_convert_output'):
    f=getattr(lib,name);f.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];f.restype=ctypes.c_int
def check(values,source_dtype,target_dtype,name,badcode):
    src=values.view(source_dtype);expected=src.to(target_dtype);out=torch.empty(src.shape,dtype=target_dtype)
    rc=getattr(lib,name)(src.data_ptr(),out.data_ptr(),src.numel())
    finite=~expected.isnan();assert torch.equal(out.view(torch.int16)[finite],expected.view(torch.int16)[finite])
    assert torch.equal(out.isnan(),expected.isnan());assert rc==badcode,(name,rc,badcode)
rows=[]
for n in (1,3,7,8,9,15,16,17,65536):
    values=torch.arange(65536,dtype=torch.int32).to(torch.int16) if n==65536 else torch.randint(-32768,32768,(n,),dtype=torch.int16)
    bf=values.view(torch.bfloat16).half();b=-2 if (~torch.isfinite(bf)).any() else 0
    check(values,torch.bfloat16,torch.float16,'mm_convert_input',b)
    b=-3 if ((values.to(torch.int32)&0x7fff)>=0x7bff).any() else 0
    check(values,torch.float16,torch.bfloat16,'mm_fast_convert_output',b)
    rows.append(n)
# Random ordering covers vector lanes mixing zeros/subnormal/normal and NaNs.
torch.manual_seed(821)
for _ in range(20):
    v=torch.randint(-32768,32768,(1027,),dtype=torch.int16)
    check(v,torch.bfloat16,torch.float16,'mm_convert_input',-2)
    check(v,torch.float16,torch.bfloat16,'mm_fast_convert_output',-3)
Path('/work/logs/conversion-unit.json').write_text(json.dumps({'passed':True,'sizes':rows,'exhaustive_patterns_per_dtype':65536,'random_mixed_batches':20,'comparison':'all non-NaN bits exact; NaN classification; input/output guard status exact'},indent=2)+'\n')
print('PASS exhaustive conversions',flush=True)
