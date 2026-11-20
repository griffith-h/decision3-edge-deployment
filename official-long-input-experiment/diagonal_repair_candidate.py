"""Pinned sparse-Q352 dot experiment; original numerical gates remain mandatory."""
from sparse_cpu_backend import DiagonalRepair

def install(owner):
    state=owner.medium_sparse.repair;candidate=DiagonalRepair(352)
    config=candidate.configuration.lower()
    if 'openblas 0.3.30' not in config or 'cortexa57' not in config:
        raise ValueError('Diagonal Q352 requires the qualified pinned CPU BLAS configuration')
    previous=(state.blas_repair,state.backend,state.counts['backend'])
    state.blas_repair=candidate;state.backend='diagonal-q352';state.counts['backend']='diagonal-q352'
    def close():
        if state.blas_repair is candidate:
            state.blas_repair,state.backend,state.counts['backend']=previous
    return close
