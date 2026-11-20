"""Bounded band2.5 experiment after three measured omitted medium-input points.

Only the existing empirical uncertainty coefficient changes from2e-7 to2.5e-7.
The original10%/65536-point budgets and whole-operation fallback remain. This
does not establish a universal accumulation bound or relax output parity gates.
"""
def install(owner):
    state=owner.medium_sparse.repair;previous=state.noise_factor
    missing=object();old=state.counts.get('noise_factor',missing);state.noise_factor=2.5e-7
    state.counts['noise_factor']=state.noise_factor;closed=False
    def close():
        nonlocal closed
        if closed:return
        closed=True
        if state.noise_factor==2.5e-7:
            state.noise_factor=previous
            if old is missing:state.counts.pop('noise_factor',None)
            else:state.counts['noise_factor']=old
    return close
