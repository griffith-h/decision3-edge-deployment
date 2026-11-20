"""Restore request-entry MKLDNN state after the original public request ends.

The pinned image path is allowed to execute its original fallback and complete
all subsequent computation with that original state. Only the next request is
isolated. The inference owner is serial; this is not a concurrent-server policy.
"""
import torch

def install(owner):
    model=owner.model;old=model.system_one;closed=False
    state={'calls':0,'restored':0};owner.request_backend_scope_counts=state
    def call(*args,**kwargs):
        enabled=torch.backends.mkldnn.enabled;state['calls']+=1
        try:return old(*args,**kwargs)
        finally:
            if torch.backends.mkldnn.enabled!=enabled:
                state['restored']+=1;torch.backends.mkldnn.enabled=enabled
    model.system_one=call
    def close():
        nonlocal closed
        if closed:return
        closed=True
        if model.system_one is call:model.system_one=old
    return close
