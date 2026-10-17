import ctypes,torch
from pathlib import Path
from prepared_scan import prepare
lib=ctypes.CDLL(str(Path(__file__).with_name('libchunk_scan.so')))
lib.d3_chunk_scan.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*6;lib.d3_chunk_scan.restype=ctypes.c_int

def install(block=8,tile=16):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
    original=qwen.torch_chunk_gated_delta_rule;counts={'optimized':0,'fallback':0}
    def forward(query,key,value,g,beta,chunk_size=64,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
        good=(not torch.is_grad_enabled() and all(x.device.type=='cpu' for x in (query,key,value,g,beta)) and
              query.dtype==key.dtype==value.dtype==torch.bfloat16 and g.dtype in (torch.bfloat16,torch.float32) and beta.dtype in (torch.bfloat16,torch.float32) and
              query.ndim==key.ndim==value.ndim==4 and g.ndim==beta.ndim==3 and query.shape==key.shape and query.shape[:3]==value.shape[:3] and g.shape==beta.shape==query.shape[:3] and
              query.shape[0]>0 and query.shape[2]>0 and 0<query.shape[1]<=512 and 0<query.shape[-1]<=128 and query.shape[-1]%4==0 and 0<value.shape[-1]<=128 and value.shape[-1]%4==0 and
              all(x.stride(-1)==1 for x in (query,key,value)) and initial_state is None and not output_final_state and
              all(v is None or (k=='use_cache' and v is False) for k,v in kwargs.items()))
        if not good:
            counts['fallback']+=1;return original(query,key,value,g,beta,chunk_size=chunk_size,initial_state=initial_state,output_final_state=output_final_state,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
        q,k,v=prepare(query,key,value,use_qk_l2norm_in_kernel)
        b=beta.transpose(1,2).float().contiguous();decay=g.transpose(1,2).float().contiguous().exp();out=torch.empty(v.shape,dtype=torch.bfloat16)
        rc=lib.d3_chunk_scan(*(x.data_ptr() for x in (q,k,v,decay,b,out)),q.shape[0]*q.shape[1],q.shape[2],q.shape[3],v.shape[3],block,tile)
        if rc:raise RuntimeError('validated recurrent scan rejected '+str(rc))
        counts['optimized']+=1
        return out.transpose(1,2).contiguous(),None
    qwen.torch_chunk_gated_delta_rule=forward
    def restore():qwen.torch_chunk_gated_delta_rule=original
    return restore,counts
