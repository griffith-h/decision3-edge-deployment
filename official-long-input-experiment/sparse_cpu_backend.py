"""Only gather identical BF16 row values; leave all arithmetic in TorchBMM."""
import ctypes,hashlib,json
from pathlib import Path
import torch

class RowGather:
    def __init__(self):
        root=Path(__file__).parent;manifest=json.loads((root/'sparse-cpu-build.json').read_text())
        for name,digest in manifest['sha256'].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Sparse CPU source/library mismatch')
        self.lib=ctypes.CDLL(str(root/'libsparse_cpu.so'))
        self.lib.d3_sparse_gather.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64,
            ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
        self.lib.d3_sparse_gather.restype=ctypes.c_int
    def run(self,module,inputs,positions):
        if (inputs.device.type!='cpu' or inputs.dtype!=torch.bfloat16 or inputs.ndim!=2
            or module.weight.device.type!='cpu' or module.weight.dtype!=torch.bfloat16 or module.weight.ndim!=2
            or inputs.shape[1]!=module.weight.shape[1] or inputs.stride(1)!=1 or module.weight.stride(1)!=1
            or positions.device.type!='cpu' or positions.dtype!=torch.long or positions.ndim!=2 or positions.shape[1]!=2):
            raise ValueError('Unsupported CPU gather tensors/strides')
        pairs=positions.contiguous();padded=(len(pairs)+3)//4*4
        a=torch.empty((padded,inputs.shape[1]),dtype=torch.float32);b=torch.empty_like(a)
        rc=self.lib.d3_sparse_gather(inputs.data_ptr(),inputs.stride(0),module.weight.data_ptr(),module.weight.stride(0),
          inputs.shape[0],inputs.shape[1],module.weight.shape[0],pairs.data_ptr(),len(pairs),a.data_ptr(),b.data_ptr())
        if rc:raise ValueError('Sparse CPU gather failed: '+str(rc))
        return a,b

class BlasRepair:
    """Approximate public-OpenBLAS repair; not bit-identical to every Torch GEMM."""
    def __init__(self):
        # Reuse the same source/binary verification as the exact row gather.
        self.gather=RowGather();self.lib=self.gather.lib
        self.provider=ctypes.CDLL(str(Path(torch.__file__).parent/'lib/libtorch_cpu.so'))
        self.provider.openblas_get_config.restype=ctypes.c_char_p
        self.configuration=self.provider.openblas_get_config().decode()
        if 'USE64BITINT' in self.configuration:raise ValueError('LP64 BLAS required')
        self.provider.openblas_get_num_threads.restype=ctypes.c_int
        self.gemm=self.provider.cblas_sgemm
        self.lib.d3_sparse_set_sgemm.argtypes=[ctypes.c_void_p];self.lib.d3_sparse_set_sgemm.restype=ctypes.c_int
        if self.lib.d3_sparse_set_sgemm(ctypes.cast(self.gemm,ctypes.c_void_p)):raise ValueError('Missing public BLAS')
        self.lib.d3_sparse_repair.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64,
            ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p]
        self.lib.d3_sparse_repair.restype=ctypes.c_int
    def threads(self):return self.provider.openblas_get_num_threads()
    def run(self,module,inputs,positions,output):
        if inputs.stride(1)!=1 or module.weight.stride(1)!=1 or not output.is_contiguous():raise ValueError('Unsupported BLAS repair strides')
        pairs=positions.contiguous()
        rc=self.lib.d3_sparse_repair(inputs.data_ptr(),inputs.stride(0),module.weight.data_ptr(),module.weight.stride(0),
          inputs.shape[0],inputs.shape[1],module.weight.shape[0],pairs.data_ptr(),len(pairs),output.data_ptr())
        if rc:raise ValueError('BLAS repair failed: '+str(rc))

class Preparation:
    """Bounded native norm/mask operations, validated against original Torch."""
    def __init__(self,vector_mask=False):
        self.gather=RowGather();self.lib=self.gather.lib
        self.mask_function=self.lib.d3_sparse_mask_neon if vector_mask else self.lib.d3_sparse_mask
        self.lib.d3_sparse_norms.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,
            ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p];self.lib.d3_sparse_norms.restype=ctypes.c_int
        self.lib.d3_sparse_mask.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int]+[ctypes.c_void_p]*5+[
            ctypes.c_float,ctypes.c_double,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
        self.lib.d3_sparse_mask.restype=ctypes.c_int
        self.mask_function.argtypes=self.lib.d3_sparse_mask.argtypes;self.mask_function.restype=ctypes.c_int
    def norms(self,value,need_half_norm=False):
        if value.device.type!='cpu' or value.dtype!=torch.bfloat16 or value.ndim!=2 or value.stride(1)!=1:raise ValueError('Unsupported native norm input')
        half=value.half().contiguous();norm=torch.empty(value.shape[0],dtype=torch.float32,device='cpu');loss=torch.empty_like(norm)
        hn=torch.empty_like(norm) if need_half_norm else None
        rc=self.lib.d3_sparse_norms(value.data_ptr(),value.stride(0),half.data_ptr(),value.shape[0],value.shape[1],norm.data_ptr(),loss.data_ptr(),hn.data_ptr() if hn is not None else None)
        if rc:raise ValueError('Native norm finite/range guard: '+str(rc))
        return (norm,loss,hn) if need_half_norm else (norm,loss)
    def mask(self,raw,xn,xloss,xhn,wnorm,wloss,noise,max_fraction,max_entries):
        if raw.device.type!='cpu' or raw.dtype!=torch.float32 or raw.ndim!=2 or not raw.is_contiguous():raise ValueError('Unsupported native mask input')
        for value,size in ((xn,raw.shape[0]),(xloss,raw.shape[0]),(xhn,raw.shape[0]),(wnorm,raw.shape[1]),(wloss,raw.shape[1])):
            if value.device.type!='cpu' or value.dtype!=torch.float32 or value.shape!=(size,) or not value.is_contiguous():raise ValueError('Unsupported native mask coefficient')
        output=torch.empty((max_entries,2),dtype=torch.long);count=ctypes.c_int()
        rc=self.mask_function(raw.data_ptr(),raw.shape[0],raw.shape[1],xn.data_ptr(),xloss.data_ptr(),xhn.data_ptr(),
            wnorm.data_ptr(),wloss.data_ptr(),noise,max_fraction,max_entries,output.data_ptr(),ctypes.byref(count))
        if rc==-4:raise ValueError('Sparse repair density budget exceeded')
        if rc==-3:raise ValueError('Nonfinite FP32 output')
        if rc:raise ValueError('Native mask guard failed: '+str(rc))
        return output[:count.value]

class RowBlasRepair(BlasRepair):
    """Same mask, grouped by input row; preflight must establish mode parity."""
    def __init__(self,mode=2):
        super().__init__()
        if mode not in (1,2,4):raise ValueError('Unsupported row repair mode')
        self.mode=mode
        self.lib.d3_sparse_repair_rows.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64,
            ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_int]
        self.lib.d3_sparse_repair_rows.restype=ctypes.c_int
    def run(self,module,inputs,positions,output):
        if (inputs.device.type!='cpu' or inputs.dtype!=torch.bfloat16 or inputs.ndim!=2 or inputs.stride(1)!=1
            or module.weight.device.type!='cpu' or module.weight.dtype!=torch.bfloat16 or module.weight.ndim!=2
            or module.weight.stride(1)!=1 or inputs.shape[1]!=module.weight.shape[1]
            or positions.device.type!='cpu' or positions.dtype!=torch.long or positions.ndim!=2 or positions.shape[1]!=2
            or output.device.type!='cpu' or output.dtype!=torch.bfloat16 or output.shape!=(inputs.shape[0],module.weight.shape[0])
            or not output.is_contiguous()):raise ValueError('Unsupported row BLAS repair tensors')
        pairs=positions.contiguous()
        rc=self.lib.d3_sparse_repair_rows(inputs.data_ptr(),inputs.stride(0),module.weight.data_ptr(),module.weight.stride(0),
            inputs.shape[0],inputs.shape[1],module.weight.shape[0],pairs.data_ptr(),len(pairs),output.data_ptr(),self.mode)
        if rc:raise ValueError('Row BLAS repair failed: '+str(rc))

class DiagonalRepair(RowBlasRepair):
    """Qualified pinned reduction only; independent dots, no unused cross terms."""
    def __init__(self,qblock=352):
        super().__init__(mode=2)
        if qblock not in (352,1024,16384):raise ValueError('Unsupported pinned reduction block')
        self.qblock=qblock
        self.lib.d3_sparse_repair_diagonal.argtypes=self.lib.d3_sparse_repair_rows.argtypes
        self.lib.d3_sparse_repair_diagonal.restype=ctypes.c_int
    def run(self,module,inputs,positions,output):
        # Same typed/stride validation as the independent row interface.
        if (inputs.device.type!='cpu' or inputs.dtype!=torch.bfloat16 or inputs.ndim!=2 or inputs.stride(1)!=1
            or module.weight.device.type!='cpu' or module.weight.dtype!=torch.bfloat16 or module.weight.ndim!=2
            or module.weight.stride(1)!=1 or inputs.shape[1]!=module.weight.shape[1]
            or positions.device.type!='cpu' or positions.dtype!=torch.long or positions.ndim!=2 or positions.shape[1]!=2
            or output.device.type!='cpu' or output.dtype!=torch.bfloat16 or output.shape!=(inputs.shape[0],module.weight.shape[0])
            or not output.is_contiguous()):raise ValueError('Unsupported diagonal repair tensors')
        pairs=positions.contiguous()
        rc=self.lib.d3_sparse_repair_diagonal(inputs.data_ptr(),inputs.stride(0),module.weight.data_ptr(),module.weight.stride(0),
            inputs.shape[0],inputs.shape[1],module.weight.shape[0],pairs.data_ptr(),len(pairs),output.data_ptr(),self.qblock)
        if rc:raise ValueError('Diagonal repair failed: '+str(rc))
