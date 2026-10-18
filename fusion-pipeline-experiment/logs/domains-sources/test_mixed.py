import sys
sys.path.insert(0,'/tail')
from runner import torch,json
from pathlib import Path
import ctypes
p=Path('/experiment');lib=ctypes.CDLL('/experiment/libmixed_bridge.so')
lib.mm_error.restype=ctypes.c_char_p
lib.mm_set_output_precision.argtypes=[ctypes.c_int];lib.mm_set_output_precision.restype=ctypes.c_int
lib.mm_create_core.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int];lib.mm_create_core.restype=ctypes.c_void_p
lib.mm_destroy.argtypes=[ctypes.c_void_p]
lib.mm_run_bf16.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];lib.mm_run_bf16.restype=ctypes.c_int
lib.mm_prepare_bf16.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];lib.mm_prepare_bf16.restype=ctypes.c_int
lib.mm_run_prepared_scatter_bf16.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]+[ctypes.POINTER(ctypes.c_int)]*3+[ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p];lib.mm_run_prepared_scatter_bf16.restype=ctypes.c_int
handles=[];r={'records':[],'passed':False};torch.manual_seed(7)
try:
 with torch.inference_mode():
  w=(torch.randn(32,32)*.1).bfloat16();packed=w.half().t().contiguous()
  for fp32 in (0,1):
   lib.mm_set_output_precision(fp32);h=lib.mm_create_core(32,32,packed.data_ptr(),1);assert h,lib.mm_error();handles.append(h)
   for rows in (1,31,128,615,640):
    x=torch.randn(rows,32).bfloat16();out=torch.empty(rows,32,dtype=torch.bfloat16);rc=lib.mm_run_bf16(h,rows,x.data_ptr(),None,out.data_ptr());assert rc==0,lib.mm_error();expected=(x.float()@w.float().t()).bfloat16();delta=float((out.float()-expected.float()).abs().max());assert torch.allclose(out.float(),expected.float(),atol=.008,rtol=.008)
    first=torch.empty(rows,16,dtype=torch.bfloat16);second=torch.empty_like(first);ints=ctypes.c_int*2;ptrs=ctypes.c_void_p*2;assert lib.mm_prepare_bf16(h,rows,x.data_ptr())==0;assert lib.mm_run_prepared_scatter_bf16(h,rows,2,ints(0,16),ints(16,16),ints(16,16),ptrs(first.data_ptr(),second.data_ptr()),None)==0;assert torch.equal(torch.cat((first,second),1),out)
    r['records'].append({'fp32':bool(fp32),'rows':rows,'max_hidden_delta':delta,'scatter_exact':True})
   x=torch.full((1,32),float('nan'),dtype=torch.bfloat16);out=torch.empty(1,32,dtype=torch.bfloat16);assert lib.mm_run_bf16(h,1,x.data_ptr(),None,out.data_ptr())==-2
   x=torch.full((1,32),60000,dtype=torch.bfloat16);ones=torch.ones_like(w).half().t().contiguous();lib.mm_set_output_precision(fp32);large=lib.mm_create_core(32,32,ones.data_ptr(),1);assert large;handles.append(large);assert lib.mm_run_bf16(large,1,x.data_ptr(),None,out.data_ptr())==-3
  r['finite_and_original_saturation_guards']=True
 r['passed']=True
finally:
 for h in reversed(handles):lib.mm_destroy(h)
 lib.mm_set_output_precision(0);r['destroyed']=len(handles);(p/'logs/mixed-unit.json').write_text(json.dumps(r,indent=2)+'\n')
print('MIXED UNIT PASS',r,flush=True)
