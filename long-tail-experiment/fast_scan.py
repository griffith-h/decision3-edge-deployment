"""Same long-sequence eligibility and computation; C/NEON validation replaces masks."""
import torch
from long_kernels import Kernels
import tail_native as native

class TunedKernels(Kernels):
    def __init__(self):
        self.fast_validate=False;self.scan_config={'kind':'original'}
        super().__init__()

    def scan(self, query, key, value, g, beta, chunk_size=64, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
        if not self.fast_validate:
            return super().scan(query,key,value,g,beta,chunk_size=chunk_size,initial_state=initial_state,output_final_state=output_final_state,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs)
        xs=(query,key,value,g,beta)
        length=query.shape[1] if query.ndim==4 else 0
        good=(
            not torch.is_grad_enabled() and all(x.device.type=='cpu' and x.layout==torch.strided for x in xs) and
            query.dtype==key.dtype==value.dtype==torch.bfloat16 and g.dtype in (torch.bfloat16,torch.float32) and beta.dtype in (torch.bfloat16,torch.float32) and
            query.ndim==key.ndim==value.ndim==4 and g.ndim==beta.ndim==3 and query.shape==key.shape and query.shape[:3]==value.shape[:3] and g.shape==beta.shape==query.shape[:3] and
            512<length<=2048 and 0<query.shape[0]<=8 and query.shape[2]>0 and query.shape[0]*query.shape[2]<=128 and
            0<query.shape[-1]<=128 and query.shape[-1]%4==0 and 0<value.shape[-1]<=128 and value.shape[-1]%4==0 and
            all(x.stride(-1)==1 and all(s>=0 for s in x.stride()) for x in (query,key,value)) and
            initial_state is None and not output_final_state and type(chunk_size) is int and chunk_size>0 and
            all(v is None or (k=='use_cache' and v is False) for k,v in kwargs.items()))
        def execute():
            from prepared_scan import prepare
            if not native.validate_raw(*xs):raise ValueError('nonfinite or non-model scan domain')
            q,k,v=prepare(query,key,value,use_qk_l2norm_in_kernel)
            b=beta.transpose(1,2).float().contiguous();decay=g.transpose(1,2).float().contiguous().exp()
            if not all(native.finite(x) for x in (q,k,v,decay)):raise ValueError('nonfinite prepared scan')
            return native.scan(q,k,v,decay,b,**self.scan_config).transpose(1,2).contiguous(),None
        return self.call('scan',good,length,execute,lambda:self.original_scan(query,key,value,g,beta,chunk_size=chunk_size,initial_state=initial_state,output_final_state=output_final_state,use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,**kwargs))
