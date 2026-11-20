"""Fuse sparse-repair norm/mask preparation without changing its arithmetic rule."""
from sparse_cpu_backend import Preparation

def install(owner):
    state=owner.medium_sparse.repair;previous=state.preparation;candidate=Preparation();state.preparation=candidate
    def close():
        if state.preparation is candidate:state.preparation=previous
    return close
