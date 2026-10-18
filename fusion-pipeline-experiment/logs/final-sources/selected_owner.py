"""Frozen selected configuration, no service integration."""
from fp32_owner import *
class Owner(Owner):
 def use(self,mode):
  assert mode in ('current','baseline','candidate')
  if mode!='candidate':return super().use(mode)
  precise=self.execution.get('precise615',False);readout=self.execution.get('readout',False);fp32=self.execution.get('fp32',False)
  if fp32:
   domain=self.execution.get('fp32_domain','all');assert domain in ('all','mlp','projections')
   assert domain=='projections' or (precise and self.execution.get('precise_width',640)==640)
   prefix={'all':'float','mlp':'mlponly','projections':'projection'}[domain]
   return super().use(prefix+('_readout' if readout else ''))
  if precise:
   wide=self.execution.get('precise_width',512)==640
   return super().use(('precisewide' if wide else 'precise')+('_readout' if readout else ''))
  return super().use('readout' if readout else 'candidate')
