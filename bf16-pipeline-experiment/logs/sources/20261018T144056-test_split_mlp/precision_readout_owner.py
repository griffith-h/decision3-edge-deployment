from readout_owner import *
class Owner(Owner):
 def use(self,mode):
  self.precise_mlp.long_tokens=615 if mode in ('precise','precise_readout','precisewide','precisewide_readout') else 0
  self.precise_mlp.width=640 if mode.startswith('precisewide') else 512
  super().use('readout' if mode in ('precise_readout','precisewide_readout') else 'candidate' if mode in ('precise','precisewide') else mode)
