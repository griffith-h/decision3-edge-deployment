"""Experimental scan only; retain the combined bridge and all other kernels."""
import ctypes
from pathlib import Path
import torch
from cpu_attention_optimized import chunk_delta_prefill

lib=ctypes.CDLL(str(Path(__file__).with_name('libd3_scan.so')))
lib.d3_scan.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*7
lib.d3_scan.restype=ctypes.c_int

def compact(q,k,v,g,beta,use_qk_l2norm_in_kernel=False):
    """One zero-state chunk at actual token length; no padding or state work."""
    dtype=q.dtype
    q,k,v,beta,g=[x.transpose(1,2).to(torch.float32,memory_format=torch.contiguous_format)
                for x in (q,k,v,beta,g)]
    if use_qk_l2norm_in_kernel:
        q=q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)
        k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
    q=q*q.shape[-1]**-.5
    cum=g.cumsum(-1)
    upper=torch.ones(q.shape[-2],q.shape[-2],device=q.device,dtype=torch.bool).triu(1)
    pair=(cum.unsqueeze(-1)-cum.unsqueeze(-2)).masked_fill(upper,float('-inf')).exp()
    system=((k*beta.unsqueeze(-1))@k.transpose(-1,-2))*pair
    attn=(q@k.transpose(-1,-2))*pair
    values=torch.linalg.solve_triangular(system,v*beta.unsqueeze(-1),upper=False,unitriangular=True)
    return (attn@values).transpose(1,2).to(dtype,memory_format=torch.contiguous_format),None

def recurrent(q,k,v,g,beta,use_qk_l2norm_in_kernel=False,tile=32,fused=0,threads=3):
    dtype=q.dtype
    # .to(float32, memory_format=contiguous_format) may return the original
    # noncontiguous view if its dtype already matches (notably FP32 decay).
    # Every raw pointer passed to C must have an explicitly dense layout.
    q,k,v,beta,g=[x.transpose(1,2).float().contiguous()
                for x in (q,k,v,beta,g)]
    if use_qk_l2norm_in_kernel:
        q=q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)
        k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
    q=q*q.shape[-1]**-.5
    decay=g.exp()
    out=torch.empty(v.shape,dtype=dtype,device='cpu')
    rc=lib.d3_scan(*(x.data_ptr() for x in (q,k,v,decay,beta,out)),q.shape[0]*q.shape[1],
                   q.shape[2],q.shape[3],v.shape[3],tile,threads,fused)
    if rc:raise RuntimeError('d3_scan rejected validated shape: '+str(rc))
    return out.transpose(1,2).contiguous(),None

def make_scan(original,variant='recurrent',tile=32,fused=0,threads=3):
    counts={'optimized':0,'fallback':0}
    def forward(query,key,value,g,beta,chunk_size=64,initial_state=None,
                output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
        # Only the validated BF16 prefill contract; all other calls retain the
        # existing scan, including initial/final state and packed sequences.
        good=(all(x.device.type=='cpu' for x in (query,key,value,g,beta)) and
              query.dtype==key.dtype==value.dtype==torch.bfloat16 and
              g.dtype in (torch.float32,torch.bfloat16) and beta.dtype in (torch.float32,torch.bfloat16) and
              query.ndim==key.ndim==value.ndim==4 and g.ndim==beta.ndim==3 and
              query.shape==key.shape and query.shape[:3]==value.shape[:3] and
              g.shape==beta.shape==query.shape[:3] and query.shape[0]>0 and query.shape[2]>0 and
              0<query.shape[1]<=(128 if variant=='compact' else 512) and
              0<query.shape[-1]<=128 and 0<value.shape[-1]<=128 and value.shape[-1]%4==0 and
              initial_state is None and not output_final_state and
              all(v is None or (k=='use_cache' and v is False) for k,v in kwargs.items()))
        if not good:
            counts['fallback']+=1
            return original(query,key,value,g,beta,chunk_size=chunk_size,initial_state=initial_state,
                            output_final_state=output_final_state,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
        counts['optimized']+=1
        if variant=='compact':return compact(query,key,value,g,beta,use_qk_l2norm_in_kernel)
        return recurrent(query,key,value,g,beta,use_qk_l2norm_in_kernel,tile,fused,threads)
    forward.counts=counts
    return forward
