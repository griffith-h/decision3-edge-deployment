"""Reuse the Owner's FP32-output bank inside the bounded long projection scope.

Inputs/weights retain the existing BF16-to-FP16 conversion, including its range
and underflow limits. The projection output is rounded from FP32 directly to
BF16, without an FP16 output boundary. Numerical parity must still be measured.
The existing Owner owns and destroys every created handle.
"""
from long_projection_candidate import BoundedProjection


class PreciseProjection(BoundedProjection):
    def __init__(self,owner,**limits):
        self.engine=owner.engine;self.mixed=owner.mixed
        self.float_handles=owner.float_handles;self.ensure_original=self.engine.ensure
        super().__init__(owner,**limits)
        self.counts.update(fp32_created=0,fp32_keys=[],retired_half=0)
        def ensure(module):
            if not self.active:return self.ensure_original(module)
            # Packed attention/scan modules are included. Their original group
            # already passed the row tiler's weight/version/hook guards. MLP
            # uses its separately owned native contexts, not this ensure path.
            key=id(module)
            if key not in self.float_handles:
                previous=self.mixed.mm_set_output_precision(1)
                try:self.float_handles[key]=self.engine.create(module.weight,self.engine.core.get(key,0))
                finally:self.mixed.mm_set_output_precision(previous)
                self.counts['fp32_created']+=1
            if key not in self.counts['fp32_keys']:self.counts['fp32_keys'].append(key)
            return self.float_handles[key]
        self.ensure_wrapper=ensure;self.engine.ensure=ensure

    def before_scope(self):
        # Scope entry runs on the owner thread after earlier work has joined.
        # Existing FP32 and native banks are protected by Owner.retire.
        before=self.owner.cache_retired['half'];self.owner.retire('half')
        self.counts['retired_half']+=self.owner.cache_retired['half']-before

    def close(self):
        if self.closed:return
        if self.active:raise RuntimeError('Cannot close active FP32 projection scope')
        if self.engine.ensure is self.ensure_wrapper:self.engine.ensure=self.ensure_original
        super().close()


def install(owner):
    owner.long_projection=PreciseProjection(owner)
    return owner.long_projection.close
