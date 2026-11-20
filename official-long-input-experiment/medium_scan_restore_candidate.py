"""Experimental existing native scan in the guarded medium domain, CPU SDPA kept.

Only changes the medium precision scope policy. The existing native scan owns
its complete-state/range/cache/failure contracts. Real final-output numerical
and task gates must pass before this can replace the qualified CPU-scan route.
"""
def install(owner):
    state=owner.medium_precision
    if state.active:raise RuntimeError('Cannot enable native scan during an active precision scope')
    previous=state.native_scan_enabled;state.native_scan_enabled=True;closed=False
    def close():
        nonlocal closed
        if closed:return
        if state.active:raise RuntimeError('Cannot restore native scan policy during an active precision scope')
        closed=True;state.native_scan_enabled=previous
    return close
