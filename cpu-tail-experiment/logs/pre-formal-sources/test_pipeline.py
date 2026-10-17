from baseline import stats
from lut_ops import Tables,native_path
import ctypes,torch,json
from pathlib import Path
lib=ctypes.CDLL(str(native_path('libpipeline_bridge.so')));fn=lib.mm_fuse_mlp_half;fn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_size_t];fn.restype=ctypes.c_int
r={'cases':[]};torch.set_num_threads(3);torch.manual_seed(391)
with torch.inference_mode():
 t=Tables();raw=torch.arange(65536,dtype=torch.int32).short();safe=(raw.int()&0x7fff)<0x7bff;gate=raw[safe].contiguous().view(torch.float16);up=torch.full_like(gate,.25);out=torch.empty_like(gate)
 assert fn(gate.data_ptr(),up.data_ptr(),out.data_ptr(),t.silu.data_ptr(),gate.numel())==0
 ref=(torch.nn.functional.silu(gate.bfloat16())*up.bfloat16()).half();assert torch.equal(out.view(torch.int16),ref.view(torch.int16));r['cases'].append('all safe FP16 patterns with quarter up')
 for n in (1,3,4,7,8,9,2047,2048,2049,65536,115*3584):
  gate=(torch.randn(n)*4).half();up=(torch.randn(n)*2).half();out=torch.empty_like(gate);saved=(gate.clone(),up.clone())
  assert fn(gate.data_ptr(),up.data_ptr(),out.data_ptr(),t.silu.data_ptr(),n)==0
  ref=(torch.nn.functional.silu(gate.bfloat16())*up.bfloat16()).half();assert torch.equal(out.view(torch.int16),ref.view(torch.int16));assert torch.equal(gate,saved[0]) and torch.equal(up,saved[1]);r['cases'].append('random '+str(n))
 for value in (float('inf'),float('nan'),65504.):
  gate=torch.full((17,),value,dtype=torch.float16);up=torch.ones_like(gate);out=torch.empty_like(gate);assert fn(gate.data_ptr(),up.data_ptr(),out.data_ptr(),t.silu.data_ptr(),17)==-3
 gate=torch.full((17,),1000,dtype=torch.float16);up=torch.full_like(gate,1000);out=torch.empty_like(gate);assert fn(gate.data_ptr(),up.data_ptr(),out.data_ptr(),t.silu.data_ptr(),17)==-2
 r['range_checks']=4;r['passed']=True
Path('/work/logs/pipeline-unit.json').write_text(json.dumps(r,indent=2)+'\n');print('PIPELINE UNIT',len(r['cases']),flush=True)
