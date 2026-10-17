import ctypes,json,sys
import torch
sys.path[:0]=['/base','/work']
from adapter import Engine
torch.set_num_threads(3);torch.manual_seed(87)
e=Engine(fusion=True)
try:
    raw=torch.randint(0,2**32,(100003,),dtype=torch.int64).to(torch.uint32)
    f=raw.view(torch.float32);out=torch.empty(f.shape,dtype=torch.bfloat16)
    e.lib.mm_convert_output.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t]
    e.lib.mm_convert_output(f.data_ptr(),out.data_ptr(),f.numel())
    expected=f.to(torch.bfloat16)
    different=out.view(torch.int16)!=expected.view(torch.int16)
    indices=different.nonzero().flatten()[:20].tolist()
    print(json.dumps({'mismatches':int(different.sum()),'nan_mismatches':int((different&torch.isnan(f)).sum()),'finite_mismatches':int((different&torch.isfinite(f)).sum()),
                     'examples':[(hex(int(raw[i])),hex(int(out.view(torch.uint16)[i])),hex(int(expected.view(torch.uint16)[i]))) for i in indices]}),flush=True)
finally:e.close()
