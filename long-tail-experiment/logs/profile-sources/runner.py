import sys
sys.path.insert(0,'/long')
from owner import Owner as PreviousOwner
from projected_owner import state,stats,P,json,time,copy,torch
from metrics import detailed

class Owner(PreviousOwner):
    def use(self,mode):
        assert mode in ('baseline','current')
        super().use('current' if mode=='current' else 'both')
