"""Keep the known CPU BF16 visual patch fallback local to its own call.

The pinned runtime already falls back to BLAS and disables MKLDNN globally on
this unsupported ARM operation. Use that same BLAS path for this exact patched
Conv3d only, restoring the preceding backend flags even when the call fails.
No checkpoint, visual preprocessing, matrix arithmetic or text backend changes.
"""
import torch

def targets(backbone):
    found=[]
    for module in backbone.modules():
        function=getattr(module.forward,'__func__',None);proj=getattr(module,'proj',None)
        if (getattr(function,'__module__',None)=='d3_runtime' and
            getattr(function,'__name__',None)=='_linear_patch_embed_forward' and
            isinstance(proj,torch.nn.Conv3d) and proj.weight.device.type=='cpu' and
            proj.weight.dtype==torch.bfloat16 and tuple(proj.kernel_size)==tuple(proj.stride)
            and hasattr(module,'embed_dim')):found.append(module)
    return found

def install(owner):
    modules=targets(owner.model.backbone)
    if len(modules)!=1:raise RuntimeError('Expected exactly one pinned CPU BF16 visual patch embedding')
    module=modules[0];old=module.forward;closed=False
    state={'calls':0,'scoped':0};owner.visual_backend_scope_counts=state
    def forward(*args,**kwargs):
        state['calls']+=1
        if not torch.backends.mkldnn.enabled:return old(*args,**kwargs)
        state['scoped']+=1
        with torch.backends.mkldnn.flags(enabled=False):return old(*args,**kwargs)
    module.forward=forward
    def close():
        nonlocal closed
        if closed:return
        closed=True
        if module.forward is forward:module.forward=old
    return close
