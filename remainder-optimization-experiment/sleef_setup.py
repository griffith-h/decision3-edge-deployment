"""Use the public SLEEF function already exported by installed open Torch."""
import ctypes
from pathlib import Path
import torch
_library=None
def prepare():
    global _library
    import cpu_math
    _library=ctypes.CDLL(str(Path(torch.__file__).parent/'lib/libtorch_cpu.so'))
    function=_library.Sleef_expf4_u10advsimd
    lib=ctypes.CDLL(str(Path(__file__).with_name('libcpu_math_sleef.so')))
    lib.d3_activation.argtypes=cpu_math.lib.d3_activation.argtypes
    lib.d3_norm.argtypes=cpu_math.lib.d3_norm.argtypes
    lib.d3_set_exp.argtypes=[ctypes.c_void_p]
    lib.d3_set_exp(ctypes.cast(function,ctypes.c_void_p))
    cpu_math.lib=lib
    return lib
