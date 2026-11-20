"""Development-only selected FP32 dots for the original enabled CPU backend.

Use the existing bounded native dots with one continuous K reduction. The
16384 block exceeds every accepted K (<=10240), so no partial sum is restarted.
No input/model/budget/gate changes. Full-model qualification is still required.
"""
import torch
from sparse_cpu_backend import DiagonalRepair

def original_backend():
    return (torch.backends.mkldnn.enabled and torch.get_num_threads()==3
            and torch.get_float32_matmul_precision()=='highest')

class EnabledSequentialRepair(DiagonalRepair):
    def __init__(self):
        if not original_backend():raise ValueError('Original enabled CPU backend required')
        super().__init__(16384)
    def run(self,module,inputs,positions,output):
        if not original_backend():raise ValueError('Original enabled CPU backend changed')
        return super().run(module,inputs,positions,output)

def install(owner):
    state=owner.medium_sparse.repair;candidate=EnabledSequentialRepair()
    previous=(state.blas_repair,state.backend,state.counts['backend'])
    state.blas_repair=candidate;state.backend='sequential-enabled-cpu';state.counts['backend']=state.backend
    def close():
        if state.blas_repair is candidate:
            state.blas_repair,state.backend,state.counts['backend']=previous
    return close
