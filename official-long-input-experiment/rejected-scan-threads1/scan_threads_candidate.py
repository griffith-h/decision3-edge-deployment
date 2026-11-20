"""Experimental1-thread original CPU scan only; request/repair backend stays3."""
import torch

class ScanThreads:
    def __init__(self,owner):
        self.owner=owner;self.original=owner.long.original_scan;self.enabled=True;self.active=False;self.closed=False
        self.counts={'scopes':0,'bypass':0,'errors':[]}
        def forward(*args,**kwargs):
            if self.closed or not self.enabled or not owner.medium_precision.active:
                self.counts['bypass']+=1;return self.original(*args,**kwargs)
            if self.active or torch.get_num_threads()!=3:raise RuntimeError('Unexpected original scan thread context')
            self.active=True;self.counts['scopes']+=1
            try:
                torch.set_num_threads(1)
                return self.original(*args,**kwargs)
            except Exception as exc:
                self.counts['errors'].append(type(exc).__name__+': '+str(exc));raise
            finally:torch.set_num_threads(3);self.active=False
        self.forward=forward;owner.long.original_scan=forward
    def close(self):
        if self.closed:return
        if self.active:raise RuntimeError('Cannot close active scan thread scope')
        self.closed=True
        if self.owner.long.original_scan is self.forward:self.owner.long.original_scan=self.original

def install(owner):
    owner.scan_threads=ScanThreads(owner)
    return owner.scan_threads.close
