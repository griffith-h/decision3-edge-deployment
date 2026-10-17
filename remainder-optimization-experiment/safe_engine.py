"""FP16-output candidate with original CPU fallback outside FP16 output range."""
from adapter import Engine
class OutputRangeError(RuntimeError):pass
class SafeEngine(Engine):
    def status(self,status):
        if status==-3:raise OutputRangeError('FP16 output saturated/nonfinite')
        return super().status(status)
    def execute(self,module,x,prepared=False,output=None):
        try:return super().execute(module,x,prepared,output)
        except OutputRangeError:
            self.stats['half_output_cpu_fallbacks']=self.stats.get('half_output_cpu_fallbacks',0)+1
            self.cpu_fallbacks+=1
            # Same CPU implementation already used for unsupported rows/types;
            # original weights and complete model math, no clamping or dropping.
            return module._bp_original(x)
