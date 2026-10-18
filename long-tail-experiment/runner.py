import sys
sys.path.insert(0,'/long')
from owner import Owner as PreviousOwner
from projected_owner import state,stats,P,json,time,copy,torch
from metrics import detailed
import long_kernels as lk
import tail_native as native
from fast_scan import TunedKernels
from mlp_schedule import Scheduler
from bucket_graph import lib as graphlib
from pathlib import Path

class Owner(PreviousOwner):
    def __init__(self):
        self.scheduler=None;self.native_originals=(lk.attention,lk.scan_prepared);self.bank_original=None
        super().__init__()
        try:
            self.long.close();self.long=TunedKernels()
            self.selected_available=(P/'selected.json').exists()
            self.selected=json.loads((P/'selected.json').read_text()) if self.selected_available else {'attention':{},'scan':{'kind':'original'}}
            self.scheduler=Scheduler(self.bank,lambda g,mask:graphlib.row_graph_set_core(g.handle,mask))
            self.bank_original=self.bank.run_rows;self.bank.run_rows=self.scheduler.run
        except BaseException:self.close();raise

    def use(self,mode):
        assert mode in ('baseline','current','validate','attention','scan','mlp','combined','validate_mlp')
        if mode in ('attention','combined') and not self.selected_available:raise ValueError('Run native replay/selection first')
        super().use('current' if mode=='current' else 'both')
        self.long.fast_validate=mode in ('validate','scan','combined','validate_mlp')
        self.long.scan_config=self.selected['scan'] if mode in ('scan','combined') else {'kind':'original'}
        lk.attention=(lambda *args:native.attention(*args,**self.selected['attention'])) if mode in ('attention','combined') else self.native_originals[0]
        self.scheduler.enabled=mode in ('mlp','combined','validate_mlp');self.scheduler.reset()

    def counts(self):return {**super().counts(),'mlp_schedule':copy.deepcopy(self.scheduler.counts)}

    def close(self):
        try:
            if self.scheduler:self.scheduler.close()
        finally:
            if self.bank_original and hasattr(self,'bank'):self.bank.run_rows=self.bank_original
            lk.attention,lk.scan_prepared=self.native_originals
            super().close()
