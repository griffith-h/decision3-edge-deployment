"""Four-lane version of the same guarded FP32 uncertainty mask and row order."""
from sparse_cpu_backend import Preparation

def install(owner):
    state=owner.medium_sparse.repair;previous=state.preparation;candidate=Preparation(vector_mask=True)
    state.preparation=candidate
    state.counts['mask_backend']='NEON, same FP32 operations and positions'
    def close():
        if state.preparation is candidate:state.preparation=previous
    return close
