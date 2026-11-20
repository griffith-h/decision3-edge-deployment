"""Switch only row storage, retaining the qualified original four-lane ELF."""
from selected_layout_tail import OriginalLayout

class OriginalLayoutScope:
    def __init__(self,owner):
        self.owner=owner;self.state=owner.medium_sparse.repair
        self.old_kernel=self.state.blas_repair;self.old_request=owner.model.system_one
        self.enabled=False;self.closed=False
        def trusted(module):
            return any(module is item for modules,_ in owner.engine.packed.values() for item in modules)
        self.kernel=OriginalLayout(trusted_inference=trusted)
        def request(*args,**kwargs):
            if not self.enabled:return self.old_request(*args,**kwargs)
            self.kernel.begin()
            try:return self.old_request(*args,**kwargs)
            finally:self.kernel.end()
        self.wrapper=request;owner.model.system_one=request
    def use(self,enabled):
        if self.closed or self.kernel.active:raise RuntimeError('Cannot switch active/closed layout')
        self.enabled=bool(enabled)
        self.state.blas_repair=self.kernel if enabled else self.old_kernel
    def close(self):
        if self.closed:return
        self.use(False);self.kernel.close();self.closed=True
        if self.owner.model.system_one is self.wrapper:self.owner.model.system_one=self.old_request
