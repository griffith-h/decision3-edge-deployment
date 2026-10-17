"""Verify public MatMul core-mask behavior using a tiny exact FP16 matrix."""
import ctypes,json,time
from pathlib import Path
lib=ctypes.CDLL('/work/libmatmul_core_probe.so')
lib.mm_test_core.argtypes=[ctypes.c_int]
lib.mm_error.restype=ctypes.c_char_p
lib.mm_create.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.mm_create.restype=ctypes.c_void_p
lib.mm_run.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];lib.mm_run.restype=ctypes.c_int
lib.mm_destroy.argtypes=[ctypes.c_void_p]
a=(ctypes.c_uint16*(128*128))(*([0x3c00]*(128*128)))
b=(ctypes.c_uint16*(128*128))(*([0x3c00]*(128*128)))
c=(ctypes.c_float*(128*128))()
records=[]
for mask in (0,1,2,3):
    lib.mm_test_core(mask);handle=lib.mm_create(128,128,b)
    row={'mask':mask,'created':bool(handle)}
    try:
        if not handle:row['error']=lib.mm_error().decode()
        else:
            times=[]
            for i in range(3):
                t=time.perf_counter();status=lib.mm_run(handle,128,a,c);times.append(time.perf_counter()-t)
                if status:break
                assert all(x==128.0 for x in c)
            row.update(run_status=status,seconds=times,outputs_exact=not status)
            if status:row['error']=lib.mm_error().decode()
    finally:
        if handle:lib.mm_destroy(handle)
    records.append(row);print(json.dumps(row),flush=True)
Path('/work/logs/core-masks.json').write_text(json.dumps({'scope':'Public RKNN 2.5.0 MatMul API on RK3576, M=K=N=128, half ones -> FP32 128. This is a capability check, not a model throughput benchmark.','records':records},indent=2)+'\n')
