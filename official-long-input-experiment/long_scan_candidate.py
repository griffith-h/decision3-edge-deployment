"""Full state-carry scan for 2048<T<=8192, bounded preparation/coefficients."""
import ctypes,hashlib,json
from pathlib import Path
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
import long_kernels as lk
import tail_native as native
from prepared_scan import prepare

root=Path(__file__).resolve().parent
build=json.loads((root/'scan-build.json').read_text())
for name,digest in build['sha256'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise RuntimeError('Extended scan source/binary identity mismatch: '+name)
lib=ctypes.CDLL(str(root/'liblong_scan_extended.so'))
for name,fn in (('d3_set_sgemm',lk.blas.scipy_cblas_sgemm64_),('d3_set_strsm',lk.blas.scipy_cblas_strsm64_)):
    getattr(lib,name).argtypes=[ctypes.c_void_p];getattr(lib,name)(ctypes.cast(fn,ctypes.c_void_p))
lib.d3_chunk_scan.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*6;lib.d3_chunk_scan.restype=ctypes.c_int

def supported(query,key,value,g,beta):
    xs=(query,key,value,g,beta)
    if any(not isinstance(x,torch.Tensor) for x in xs):return False
    if not (query.ndim==key.ndim==value.ndim==4 and g.ndim==beta.ndim==3):return False
    b,t,h,kd=query.shape;vd=value.shape[-1]
    return (not torch.is_grad_enabled() and all(x.device.type=='cpu' and x.layout==torch.strided for x in xs)
        and query.dtype==key.dtype==value.dtype==torch.bfloat16
        and g.dtype in (torch.bfloat16,torch.float32) and beta.dtype in (torch.bfloat16,torch.float32)
        and query.shape==key.shape and query.shape[:3]==value.shape[:3] and g.shape==beta.shape==query.shape[:3]
        and 2048<t<=8192 and 0<b<=8 and h>0 and b*h<=128
        and 0<kd<=128 and kd%4==0 and 0<vd<=128 and vd%4==0
        and b*h*t*(2*kd+vd)*4<=256*1024*1024
        and b*h*((t+31)//32)*(2*32*32+2*32)*4<=128*1024*1024
        and all(x.stride(-1)==1 and all(s>=0 for s in x.stride()) for x in (query,key,value)))

def kernel(query,key,value,g,beta,normalize):
    if not supported(query,key,value,g,beta):raise ValueError('Unsupported extended scan contract')
    if not all(bool(torch.isfinite(x).all()) for x in (query,key,value,g,beta)) or bool((g>0).any()) or bool(((beta<0)|(beta>1)).any()):
        raise ValueError('Nonfinite or non-model scan domain')
    q,k,v=prepare(query,key,value,normalize)
    decay=g.transpose(1,2).float().contiguous().exp();bb=beta.transpose(1,2).float().contiguous()
    if not all(native.finite(x) for x in (q,k,v,decay)):raise ValueError('Nonfinite prepared scan')
    out=torch.empty(v.shape,dtype=torch.bfloat16)
    rc=lib.d3_chunk_scan(*(x.data_ptr() for x in (q,k,v,decay,bb,out)),q.shape[0]*q.shape[1],q.shape[2],q.shape[3],v.shape[3],32,64)
    if rc or not native.finite(out):raise ValueError('Extended scan rejected '+str(rc))
    return out.transpose(1,2).contiguous(),None

class LongScan:
    def __init__(self):
        self.previous=qwen.torch_chunk_gated_delta_rule;self.closed=False;self.disabled=False
        self.counts={'optimized':0,'bypass':0,'fallback':0,'errors':[],'lengths':{}}
        qwen.torch_chunk_gated_delta_rule=self.forward
    def forward(self,query,key,value,g,beta,chunk_size=64,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
        good=(not self.closed and not self.disabled and initial_state is None and not output_final_state
            and type(chunk_size) is int and chunk_size>0 and all(v is None or (k=='use_cache' and v is False) for k,v in kwargs.items())
            and supported(query,key,value,g,beta))
        if good:
            try:result=kernel(query,key,value,g,beta,use_qk_l2norm_in_kernel)
            except (ValueError,RuntimeError) as exc:
                self.counts['fallback']+=1;self.counts['errors'].append(str(exc));self.disabled=True
            else:
                self.counts['optimized']+=1;t=str(query.shape[1]);self.counts['lengths'][t]=self.counts['lengths'].get(t,0)+1
                return result
        else:self.counts['bypass']+=1
        return self.previous(query,key,value,g,beta,chunk_size=chunk_size,initial_state=initial_state,
            output_final_state=output_final_state,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
    def close(self):
        if self.closed:return
        self.closed=True
        if qwen.torch_chunk_gated_delta_rule==self.forward:qwen.torch_chunk_gated_delta_rule=self.previous

def install(owner):
    owner.long_scan=LongScan()
    return owner.long_scan.close
