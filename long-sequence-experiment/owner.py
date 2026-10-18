import sys
sys.path.insert(0, '/row')
from fused_owner import Owner as PreviousOwner
from projected_owner import state, stats, P, json, time, copy, torch
from metrics import detailed
from long_kernels import Kernels

class Owner(PreviousOwner):
    def __init__(self):
        self.long = None
        super().__init__()
        try:
            self.long = Kernels()
        except BaseException:
            self.close()
            raise

    def use(self, mode):
        kinds = {'baseline': set(), 'attention': {'attention'}, 'scan': {'scan'}, 'both': {'attention', 'scan'}, 'current': set()}
        assert mode in kinds
        super().use('current' if mode == 'current' else 'fused')
        self.long.enabled = kinds[mode]
        self.long.reset()

    def counts(self):
        return {**super().counts(), 'long_kernels': copy.deepcopy(self.long.counts)}

    def close(self):
        if self.long:
            self.long.close()
        super().close()
