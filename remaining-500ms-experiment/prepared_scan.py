import ctypes,torch
from pathlib import Path
from scan import lib as scanlib
lib=ctypes.CDLL(str(Path(__file__).with_name('libscan_prepare.so')))
fn=lib.d3_prepare_scan;fn.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int]*6;fn.restype=ctypes.c_int

def prepare(q,k,v,normalize):
    batch,length,heads,kd=q.shape;vd=v.shape[-1]
    qo=torch.empty((batch,heads,length,kd),dtype=torch.float32);ko=torch.empty_like(qo);vo=torch.empty((batch,heads,length,vd),dtype=torch.float32)
    strides=(ctypes.c_int64*9)(*[s for x in (q,k,v) for s in x.stride()[:3]])
    rc=fn(*(x.data_ptr() for x in (q,k,v,qo,ko,vo)),ctypes.addressof(strides),batch,length,heads,kd,vd,int(normalize))
    if rc:raise RuntimeError('validated scan preparation rejected '+str(rc))
    return qo,ko,vo

def install():
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
        rc=scanlib.d3_scan(*(x.data_ptr() for x in (q,k,v,decay,b,out)),q.shape[0]*q.shape[1],q.shape[2],q.shape[3],v.shape[3],16,3,5)
        if rc:raise RuntimeError('validated recurrent scan rejected '+str(rc))
        counts['optimized']+=1
        return out.transpose(1,2).contiguous(),None
    qwen.torch_chunk_gated_delta_rule=forward
    def restore():qwen.torch_chunk_gated_delta_rule=original
    return restore,counts
