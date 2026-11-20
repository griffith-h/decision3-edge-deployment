"""Scope the existing private-output row tiler to bounded long inputs.

Only the Python full-output limit changes during a long call. Native submissions
remain <=512 rows; weights, native libraries, attention and scan caps are intact.
"""
from types import MethodType


class BoundedProjection:
    def __init__(self,owner,max_rows=8192,max_output_bytes=128*1024*1024,max_length=8192):
        self.owner=owner;self.model=owner.model;self.rows=owner.projections
        self.max_rows=max_rows;self.max_output_bytes=max_output_bytes;self.max_length=max_length
        self.original=self.model.logits;self.original_good=self.rows.good
        self.closed=False;self.active=False
        self.counts={'scopes':0,'output_budget_fallbacks':0,
                     'limits':{'aggregate_rows':max_rows,'sequence_length':max_length,
                               'private_output_bytes':max_output_bytes,'native_rows':512}}
        def good(members,x):
            if not self.original_good(members,x):return False
            n=x.numel()//x.shape[-1]
            if self.active and n>2048 and n*sum(m.out_features for m in members)*x.element_size()>self.max_output_bytes:
                self.counts['output_budget_fallbacks']+=1;return False
            return True
        self.good=good;self.rows.good=good
        def scope(model,sequences,counts):
            n=len(sequences)*max(map(len,sequences),default=0)
            if (self.closed or self.active or not 2048<n<=self.max_rows
                    or max(map(len,sequences),default=0)>self.max_length
                    or not 0<len(sequences)<=8 or self.rows.max_rows!=2048
                    or self.rows.mode!='chunked'):
                return self.original(sequences,counts)
            previous=(self.rows.max_rows,self.rows.width,self.owner.fused)
            self.active=True;self.counts['scopes']+=1
            self.rows.max_rows=self.max_rows;self.rows.width=512
            # Do not incidentally widen the separate fused attention preparation
            # facade when widening projection eligibility. Its old long path
            # was ineligible and remains unchanged here.
            self.owner.fused=False
            try:
                self.before_scope()
                return self.original(sequences,counts)
            finally:
                self.rows.max_rows,self.rows.width,self.owner.fused=previous
                self.active=False
        self.wrapper=MethodType(scope,self.model);self.model.logits=self.wrapper

    def before_scope(self):pass

    def close(self):
        if self.closed:return
        if self.active:raise RuntimeError('Cannot close active projection scope')
        self.closed=True
        if self.model.logits==self.wrapper:self.model.logits=self.original
        if self.rows.good is self.good:self.rows.good=self.original_good


def install(owner):
    owner.long_projection=BoundedProjection(owner)
    return owner.long_projection.close
