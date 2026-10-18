from extended_owner import Owner as PreviousOwner,state,stats,P,json,time,copy,torch,detailed
from npu_attention import NpuAttention
class Owner(PreviousOwner):
 def __init__(self):
  self.grouped=None;self.ungrouped=None
  super().__init__();self.ungrouped=self.native_attention
  try:self.grouped=NpuAttention(fused=True,grouped=True)
  except BaseException:self.close();raise
 def use(self,mode):
  self.native_attention=self.grouped if mode=='grouped' else self.ungrouped
  super().use('wide' if mode=='grouped' else mode)
 def close(self):
  try:
   if self.grouped:self.grouped.close()
  finally:
   if self.ungrouped:self.native_attention=self.ungrouped
   super().close()
