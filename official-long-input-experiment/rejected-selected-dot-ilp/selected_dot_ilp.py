"""Own optional selected-dot kernel; original mask and repair policy stay intact."""
import ctypes,hashlib,json
from pathlib import Path
import torch
from enabled_sequential_repair_candidate import original_backend

class SelectedDot:
    def __init__(self,width):
        if width not in (4,8,16):raise ValueError('Unsupported independent-output width')
        self.width=width;root=Path(__file__).parent;build=json.loads((root/'selected-dot-build.json').read_text())
        for name,value in build['sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=value:raise ValueError('Selected dot build identity mismatch')
        self.lib=ctypes.CDLL(str(root/'libselected_dot_ilp.so'));self.fn=self.lib.d3_selected_dot_ilp
        self.fn.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64,
            ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
        self.fn.restype=ctypes.c_int;self.calls=0;self.selected=0
    def run(self,module,inputs,positions,output,debug=None):
        if not original_backend():raise ValueError('Original enabled CPU backend changed')
        if (inputs.device.type!='cpu' or inputs.dtype!=torch.bfloat16 or inputs.ndim!=2 or inputs.stride(1)!=1
            or module.weight.device.type!='cpu' or module.weight.dtype!=torch.bfloat16 or module.weight.ndim!=2
            or module.weight.stride(1)!=1 or inputs.shape[1]!=module.weight.shape[1]
            or positions.device.type!='cpu' or positions.dtype!=torch.long or positions.ndim!=2 or positions.shape[1]!=2
            or output.device.type!='cpu' or output.dtype!=torch.bfloat16 or output.shape!=(inputs.shape[0],module.weight.shape[0])
            or not output.is_contiguous()):raise ValueError('Unsupported selected dot tensors')
        if debug is not None and (debug.device.type!='cpu' or debug.dtype!=torch.float32 or debug.shape!=(len(positions),) or not debug.is_contiguous()):raise ValueError('Unsupported selected dot debug output')
        pairs=positions.contiguous()
        rc=self.fn(inputs.data_ptr(),inputs.stride(0),module.weight.data_ptr(),module.weight.stride(0),
            inputs.shape[0],inputs.shape[1],module.weight.shape[0],pairs.data_ptr(),len(pairs),output.data_ptr(),self.width,
            debug.data_ptr() if debug is not None else None)
        if rc:raise ValueError('Selected dot guard failed: '+str(rc))
        self.calls+=1;self.selected+=len(pairs)
