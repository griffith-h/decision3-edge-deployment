"""Experimental r012 composition; rejected on later coverage, not a release factory.

This prototype still includes BatchPlanner and must not be connected to
evaluation or serving as the final engine. Select a qualified composition and
validate its frozen factory on the board before integration. No model arithmetic
lives here.
"""
import copy
import importlib

MODULES=('long_projection_fp32_candidate','long_mlp_candidate',
         'long_readout_candidate','batch_candidate','long_scan_candidate',
         'long_npu_attention_candidate')
COUNTERS=('long_projection','long_mlp','long_readout','batch_planner',
          'long_scan','long_npu_attention')

class Owner:
    def __init__(self,base_factory=None,installers=None):
        self.base=None;self.restores=[];self.closed=False
        try:
            if base_factory is None:
                from accepted_owner import Owner as BaseOwner
                base_factory=BaseOwner
            self.base=base_factory();self.base.use('candidate')
            if installers is None:
                installers=[importlib.import_module(name).install for name in MODULES]
            for install in installers:self.restores.append(install(self.base))
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup_error:raise error from cleanup_error
            raise
    def __getattr__(self,name):
        return getattr(self.base,name)
    def use(self,mode):
        if self.closed or mode!='candidate':raise ValueError('Frozen long-input Owner supports candidate mode only')
    def counts(self):
        return {**self.base.counts(),'long_input':copy.deepcopy({
            name:getattr(getattr(self.base,name,None),'counts',{}) for name in COUNTERS})}
    def close(self):
        if self.closed:return
        self.closed=True;failures=[]
        for restore in reversed(self.restores):
            try:restore()
            except BaseException as exc:failures.append(exc)
        if self.base is not None:
            try:self.base.close()
            except BaseException as exc:failures.append(exc)
        if failures:raise failures[0]
