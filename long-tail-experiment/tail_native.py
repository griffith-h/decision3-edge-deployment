import ctypes
from pathlib import Path
import torch
import long_kernels as lk
P=Path(__file__).resolve().parent
att=ctypes.CDLL(str(P/'libtiled_attention.so'))
att.d3_set_sgemm.argtypes=[ctypes.c_void_p];att.d3_set_sgemm(ctypes.cast(lk.blas.scipy_cblas_sgemm64_,ctypes.c_void_p))
att.d3_attention_tiled.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*5+[ctypes.c_float]+[ctypes.c_int]*3;att.d3_attention_tiled.restype=ctypes.c_int
scans={}
for n in (2,3):
    lib=ctypes.CDLL(str(P/f'libtuned_scan{n}.so'))
    lib.d3_set_sgemm.argtypes=[ctypes.c_void_p];lib.d3_set_sgemm(ctypes.cast(lk.blas.scipy_cblas_sgemm64_,ctypes.c_void_p))
    lib.d3_set_strsm.argtypes=[ctypes.c_void_p];lib.d3_set_strsm(ctypes.cast(lk.blas.scipy_cblas_strsm64_,ctypes.c_void_p))
    lib.d3_chunk_scan.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*6;lib.d3_chunk_scan.restype=ctypes.c_int;scans[n]=lib
rec=ctypes.CDLL(str(P/'librecurrent_scan.so'));rec.d3_scan_fast.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*6;rec.d3_scan_fast.restype=ctypes.c_int
valid=ctypes.CDLL(str(P/'libvalidity.so'))
for name in ('d3_finite_bf16','d3_finite_f32'):
    getattr(valid,name).argtypes=[ctypes.c_void_p,ctypes.c_int64];getattr(valid,name).restype=ctypes.c_int
valid.d3_validate_scan.argtypes=[ctypes.c_void_p]*6+[ctypes.c_int]*7;valid.d3_validate_scan.restype=ctypes.c_int

def finite(x):
    if x.device.type!='cpu' or not x.is_contiguous() or x.dtype not in (torch.float32,torch.bfloat16):raise ValueError('Unsupported native finite signature')
    fn=valid.d3_finite_f32 if x.dtype==torch.float32 else valid.d3_finite_bf16
    return fn(x.data_ptr(),x.numel())==0

def validate_raw(q,k,v,g,beta):
    b,t,h,kd=q.shape;vd=v.shape[-1]
    s=(ctypes.c_int64*15)(*[a for x in (q,k,v,g,beta) for a in x.stride()[:3]])
    return valid.d3_validate_scan(*(x.data_ptr() for x in (q,k,v,g,beta)),ctypes.addressof(s),b,t,h,kd,vd,int(g.dtype==torch.float32),int(beta.dtype==torch.float32))==0

def attention(q,k,v,mask,scale,tile=64,threads=3,reciprocal=1):
    b,h,t,d=q.shape;out=torch.empty((b,t,h,d),dtype=q.dtype)
    strides=(ctypes.c_int64*9)(*[a for x in (q,k,v) for a in x.stride()[:3]])
    rc=att.d3_attention_tiled(*(x.data_ptr() for x in (q,k,v)),mask.data_ptr(),out.data_ptr(),ctypes.addressof(strides),b,h,k.shape[1],t,d,scale,tile,threads,reciprocal)
    if rc:raise ValueError('Tiled attention rejected '+str(rc))
    return out

def scan(q,k,v,decay,beta,kind='block',block=32,tile=64,threads=3,mode=7):
    out=torch.empty(v.shape,dtype=torch.bfloat16);args=[x.data_ptr() for x in (q,k,v,decay,beta,out)];shape=(q.shape[0]*q.shape[1],q.shape[2],q.shape[3],v.shape[3])
    if kind in ('block','original'):
        lib=lk.scan if kind=='original' else scans[threads]
        rc=lib.d3_chunk_scan(*args,*shape,block,tile)
    else:rc=rec.d3_scan_fast(*args,*shape,mode,threads)
    if rc or not finite(out):raise ValueError('Tuned scan rejected '+str(rc))
    return out
