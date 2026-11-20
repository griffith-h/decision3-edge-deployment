"""Original CPU attention and, by default, scan in the guarded medium domain.

Only the backend changes. Original B/T, padding, complete inputs, model weights
and BF16 boundaries remain. The known B1/T615 and shorter domains are preserved.
An explicit experimental policy can retain the existing native scan only.
This is a development precision repair, not a qualified release.
"""
from types import MethodType

class MediumPrecision:
    def __init__(self,owner):
        self.owner=owner;self.model=owner.model;self.original=self.model.logits
        self.closed=False;self.active=False;self.native_scan_enabled=False
        self.counts={'scopes':0,'bypass':0,'batch_shapes':{},'native_scan_scopes':0}
        def scope(model,sequences,counts):
            batch=len(sequences);length=max(map(len,sequences),default=0)
            if self.closed or self.active or not (0<batch<=8 and 512<length<=2048) or (batch,length)==(1,615):
                self.counts['bypass']+=1;return self.original(sequences,counts)
            enabled=owner.long.enabled;self.active=True
            disabled={'attention'} if self.native_scan_enabled else {'scan','attention'}
            owner.long.enabled=set(enabled)-disabled
            if self.native_scan_enabled:self.counts['native_scan_scopes']+=1
            self.counts['scopes']+=1;key=f'{batch}x{length}'
            self.counts['batch_shapes'][key]=self.counts['batch_shapes'].get(key,0)+1
            try:return self.original(sequences,counts)
            finally:owner.long.enabled=enabled;self.active=False
        self.wrapper=MethodType(scope,self.model);self.model.logits=self.wrapper
    def close(self):
        if self.closed:return
        if self.active:raise RuntimeError('Cannot close an active precision scope')
        self.closed=True
        if self.model.logits==self.wrapper:self.model.logits=self.original

def install(owner):
    owner.medium_precision=MediumPrecision(owner)
    return owner.medium_precision.close
