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
    from collections import defaultdict
    import os
    # Same contexts, but no slow Python output scan between timed trials.
    # Rotate paired trial order to avoid case-order / DVFS bias.
    chosen=[('serial_split',[by_mask[1][0],by_mask[2][0]],False),
            ('parallel_split',[by_mask[1][0],by_mask[2][0]],True),
            ('parallel_same_core',by_mask[1],True)]
    records=defaultdict(list)
    frequencies=[p/'cur_freq' for p in Path('/sys/class/devfreq').glob('*') if 'npu' in p.name]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for round_index in range(35):
            order=chosen[round_index%3:]+chosen[:round_index%3]
            for name,pair,parallel in order:
                freq={str(path):path.read_text().strip() for path in frequencies}
                t=time.perf_counter()
                if parallel:
                    futures=[pool.submit(run,h,out) for h,out in zip(pair,outputs)]
                    for future in futures:future.result()
                else:
                    for h,out in zip(pair,outputs):run(h,out)
                elapsed=time.perf_counter()-t
                if round_index>=5:records[name].append({'round':round_index,'seconds':elapsed,'frequencies_before':freq})
        assert all(x==float(k) for output in outputs for x in output)
    summary={name:{'median':statistics.median(x['seconds'] for x in rows),'min':min(x['seconds'] for x in rows),'max':max(x['seconds'] for x in rows)} for name,rows in records.items()}
    paired=[records['serial_split'][i]['seconds']/records['parallel_split'][i]['seconds'] for i in range(30)]
    result={'passed':True,'scope':'Interleaved two-MatMul probe, real M115 K1024 N3584. CPU 4-7. Five warmup rounds, 30 timed rounds per variant, rotating order. No Python output scanning between rounds. Not a d3 model speedup.',
            'affinity':sorted(os.sched_getaffinity(0)),'records':dict(records),'summary':summary,
            'median_paired_serial_over_parallel_speedup':statistics.median(paired),'parallel_faster_rounds':sum(v>1 for v in paired),'outputs_exact':True}
    Path('/work/logs/parallel-interleaved.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
finally:
    for handle in handles:lib.mm_destroy(handle)
