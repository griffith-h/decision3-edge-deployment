"""Empirical sparse band2 experiment after a measured omitted BF16 boundary.

Only the existing guarded medium sparse path uses this value. Original density,
entry, fallback, model, padding and numerical gates remain unchanged. Doubling
is an experiment, not a proven universal FP32 accumulation-error bound.
"""
def install(owner):
    state=owner.medium_sparse.repair;previous=state.noise_factor
    missing=object();old=state.counts.get('noise_factor',missing);state.noise_factor=2e-7
    state.counts['noise_factor']=state.noise_factor;closed=False
    def close():
        nonlocal closed
        if closed:return
        closed=True
        if state.noise_factor==2e-7:
            state.noise_factor=previous
            if old is missing:state.counts.pop('noise_factor',None)
            else:state.counts['noise_factor']=old
    return close
