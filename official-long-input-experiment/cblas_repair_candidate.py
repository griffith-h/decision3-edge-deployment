"""Separate approximation experiment, always using the original model0.01 gate.

Public CBLAS is not bit-identical to TorchGEMM (r027 negative evidence retained).
This variant tests whether fast sparse correction is accurate enough end to end;
passing primitive/positive tests cannot qualify it for release.
"""
from sparse_rounding import SparseRepair

def install(owner):
    previous=owner.medium_sparse.repair
    candidate=SparseRepair(backend='cblas');owner.medium_sparse.repair=candidate
    def close():
        if owner.medium_sparse.repair is candidate:owner.medium_sparse.repair=previous
        candidate.cache.clear()
    return close
