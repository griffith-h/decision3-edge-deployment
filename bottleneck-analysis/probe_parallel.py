"""Independent public MatMul contexts on two core masks; real M/K/N shape."""
from concurrent.futures import ThreadPoolExecutor
from array import array
import ctypes,json,statistics,time
from pathlib import Path
lib=ctypes.CDLL('/work/libmatmul_core_probe.so')
lib.mm_test_core.argtypes=[ctypes.c_int];lib.mm_error.restype=ctypes.c_char_p
lib.mm_create.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_void_p];lib.mm_create.restype=ctypes.c_void_p
lib.mm_run.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p];lib.mm_run.restype=ctypes.c_int
lib.mm_destroy.argtypes=[ctypes.c_void_p]
m,k,n=115,1024,3584
weights=array('H',[0x3c00])*(k*n);inputs=array('H',[0x3c00])*(m*k)
buffer_weights=(ctypes.c_uint16*len(weights)).from_buffer(weights)
buffer_inputs=(ctypes.c_uint16*len(inputs)).from_buffer(inputs)
outputs=[(ctypes.c_float*(m*n))(),(ctypes.c_float*(m*n))()]
handles=[];by_mask={}
def run(handle,output):
    status=lib.mm_run(handle,m,buffer_inputs,output)
    if status:raise RuntimeError(lib.mm_error().decode())
try:
    for mask in (0,1,2):
        pair=[]
        for _ in range(2):
            lib.mm_test_core(mask);handle=lib.mm_create(k,n,buffer_weights)
            if not handle:raise RuntimeError(lib.mm_error().decode())
            handles.append(handle);pair.append(handle)
        by_mask[mask]=pair
    cases=[('serial_same_core',by_mask[1],False),('parallel_same_core',by_mask[1],True),
           ('serial_split_core',[by_mask[1][0],by_mask[2][0]],False),
           ('parallel_split_core',[by_mask[1][0],by_mask[2][0]],True),
           ('parallel_auto',by_mask[0],True),('serial_split_recheck',[by_mask[1][0],by_mask[2][0]],False)]
    records={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for name,pair,parallel in cases:
            times=[]
            for i in range(28):
                t=time.perf_counter()
                if parallel:
                    futures=[pool.submit(run,h,out) for h,out in zip(pair,outputs)]
                    for future in futures:future.result()
                else:
                    for h,out in zip(pair,outputs):run(h,out)
                elapsed=time.perf_counter()-t
                if i>=3:times.append(elapsed)
            assert all(x==float(k) for output in outputs for x in output)
            row={'seconds':times,'median':statistics.median(times),'min':min(times),'max':max(times),'outputs_exact':True}
            records[name]=row;print(name,row['median'],flush=True)
    result={'passed':True,'scope':'Two independent M115 K1024 N3584 FP16-ones MatMul contexts, FP32 output; 3 pair warmups + 25 timed pairs per case. Public API capability/throughput test only, no d3 model parallel implementation. CPU affinity 0-3.','cases':records,
        'speedup_parallel_vs_serial_split':records['serial_split_core']['median']/records['parallel_split_core']['median'],
        'speedup_vs_serial_recheck':records['serial_split_recheck']['median']/records['parallel_split_core']['median']}
    Path('/work/logs/parallel-matmul.json').write_text(json.dumps(result,indent=2)+'\n')
finally:
    for handle in handles:lib.mm_destroy(handle)
