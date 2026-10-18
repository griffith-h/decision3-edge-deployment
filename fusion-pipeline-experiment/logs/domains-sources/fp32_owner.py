"""Restore known FP32 MatMul output only inside B1/T615 logits.

Each context fixes output type at construction. Half and FP32 handle banks are
disjoint; engine-owned handles are destroyed only through the same mixed ABI.
"""
from precision_readout_owner import *
from types import MethodType
class Owner(Owner):
 bridge_path='/experiment/libmixed_bridge.so'
 def __init__(self):
  self.float_domain="all";self.mlp_ids=set();self.float_enabled=False;self.float_active=False;self.float_handles={};self.ensure_original=None;self.scope_original=None
  super().__init__()
  try:
   self.mlp_ids={id(v) for m in self.model.backbone.modules() if type(m).__name__=='Qwen3_5MLP' for v in (m.gate_proj,m.up_proj,m.down_proj)}
   lib=self.engine.lib;lib.mm_set_output_precision.argtypes=[__import__('ctypes').c_int];lib.mm_set_output_precision.restype=__import__('ctypes').c_int
   self.ensure_original=self.engine.ensure;self.mixed=lib
   def ensure(module):
    if not self.float_active or (self.float_domain=='mlp' and id(module) not in self.mlp_ids) or (self.float_domain=='projections' and id(module) in self.mlp_ids):return self.ensure_original(module)
    key=id(module)
    if key not in self.float_handles:
     previous=self.mixed.mm_set_output_precision(1)
     try:self.float_handles[key]=self.engine.create(module.weight,self.engine.core.get(key,0))
     finally:self.mixed.mm_set_output_precision(previous)
    return self.float_handles[key]
   self.engine.ensure=ensure;self.scope_original=self.model.logits
   def scope(model,sequences,counts):
    previous=self.float_active;self.float_active=self.float_enabled and len(sequences)==1 and len(sequences[0])==615
    try:return self.scope_original(sequences,counts)
    finally:self.float_active=previous
   self.model.logits=MethodType(scope,self.model)
  except BaseException:self.close();raise
 def use(self,mode):
  self.float_enabled=mode in ('float','float_readout','mlponly','mlponly_readout','projection','projection_readout')
  self.float_domain='mlp' if mode.startswith('mlponly') else 'projections' if mode.startswith('projection') else 'all'
  if mode in ('projection','projection_readout'):return super().use('readout' if mode.endswith('_readout') else 'candidate')
  super().use('precisewide_readout' if mode in ('float_readout','mlponly_readout') else 'precisewide' if mode in ('float','mlponly') else mode)
 def counts(self):return {**super().counts(),'fp32_contexts':len(self.float_handles),'fp32_enabled':self.float_enabled,'fp32_domain':self.float_domain}
 def close(self):
  try:
   if self.scope_original and hasattr(self,'model'):self.model.logits=self.scope_original
   if self.ensure_original and hasattr(self,'engine'):self.engine.ensure=self.ensure_original
  finally:
   super().close();self.float_handles.clear()
