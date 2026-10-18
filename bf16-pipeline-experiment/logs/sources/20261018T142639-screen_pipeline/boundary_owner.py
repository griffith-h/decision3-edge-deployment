from pipeline_owner import Owner as PipelineOwner,state,stats,P,json,time,copy,torch,detailed
import ctypes
class Owner(PipelineOwner):
 bridge_path='/experiment/libfused_boundary.so'
 def use(self,mode):
  assert mode in ('current','baseline','candidate','boundary','serial')
  super().use(mode if mode!='boundary' else 'baseline')
  self.engine.lib.mm_set_mlp_fusion.argtypes=[ctypes.c_int];self.engine.lib.mm_set_mlp_fusion(1 if mode in ('boundary','candidate') else 0)
