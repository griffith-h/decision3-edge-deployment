"""Consolidate measured costs without mixing inclusive/exclusive or profiled/raw times."""
import datetime,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
def read(name):return json.loads((root/name).read_text())
p=read('logs/profile-current.json');s=read('logs/scan-operators.json')
c=read('logs/core-masks.json');parallel=read('logs/parallel-interleaved.json')
service=read('service-preservation-final.json');initial_service=read('service-preservation.json')
assert p['passed'] and s['passed'] and parallel['passed'] and service['passed']
assert p['source_sha256']=={name:service['source_sha256'][name] for name in p['source_sha256']}
assert all(r['created'] and r['outputs_exact'] for r in c['records'] if r['mask'] in (0,1,2))
assert not next(r for r in c['records'] if r['mask']==3)['created']
records=p['instrumented']
mean=lambda values:statistics.mean(values)
exclusive={key:mean(r['timings'][key]['exclusive'] for r in records) for key in records[0]['timings']}
inclusive={key:mean(r['timings'][key]['inclusive'] for r in records) for key in records[0]['timings']}
native={key:mean(r['native_bridge'][key] for r in records) for key in records[0]['native_bridge']}
total=mean(r['seconds'] for r in records)
phases={'rknn_runtime_wall':native['rknn_matmul_run'],
        'bridge_copy_shape_and_ffi':inclusive['adapter.bridge_ffi']-native['rknn_matmul_run'],
        'adapter_conversion_validation_allocation_and_output':inclusive['linear.rknn']-inclusive['adapter.bridge_ffi'],
        'gated_delta_scan_including_triangular_solve':inclusive['scan'],
        'causal_conv_and_rmsnorm':exclusive['causal_conv']+exclusive['rmsnorm'],
        'attention_sdpa_rope_and_dispatch':sum(exclusive[k] for k in ['sdpa','rotary','attention.elementwise_and_dispatch','module.other.Qwen3_5TextRotaryEmbedding'])}
phases['remaining_request_and_cpu']=total-sum(phases.values())
assert abs(sum(phases.values())-total)<1e-9
scanops={}
for row in s['cases']:
    for key,v in row['operators'].items():
        item=scanops.setdefault(key,{'seconds':0.,'calls':0})
        item['seconds']+=v['seconds'];item['calls']+=v['calls']
shim=p['clock_shim_uninstrumented']['median']
result={'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,
 'scope':'Diagnosis only. Deployed model weights, 12 layers, attention semantics, three-thread server and settings unchanged. Fixed 115-token single-question input, concurrency 1. No response cache, no reverse engineering.',
 'raw_model_seconds':{key:{k:p[key][k] for k in ['median','min','max']} for key in ['uninstrumented_before','clock_shim_uninstrumented','uninstrumented_after']},
 'instrumented_wall_mean_seconds':total,'instrumentation_caution':'Hooked attribution is about 900ms vs unhooked about 851-869ms. Do not present these phase figures as an exact partition of the earlier 859ms LAN sample. Runtime wall includes vendor SDK scheduling/synchronization, not pure NPU device compute.',
 'partition_exclusive_seconds':phases,'exclusive_details_seconds':exclusive,'native_bridge_seconds':native,
 'instrumented_overhead_vs_clock_shim_median_fraction':total/shim-1,
 'scan_microprofile':{'sum_six_unprofiled_medians':sum(row['median'] for row in s['cases']),
                     'sum_six_profiled_seconds':sum(row['profiled_seconds'] for row in s['cases']),
                     'operators':dict(sorted(scanops.items(),key=lambda x:x[1]['seconds'],reverse=True)),
                     'caution':'TorchDispatch adds about 76ms across six scan traces. Operator figures are localization evidence, not an additive raw request-time budget. All six outputs bitwise equal.'},
 'core_capability':{'mask_0_1_2_correct':True,'combined_mask_3_rejected':True,
                    'interleaved_parallel_probe':{k:v for k,v in parallel.items() if k!='records'},
                    'incremental_split_core_over_same_core_parallel_speedup':parallel['summary']['parallel_same_core']['median']/parallel['summary']['parallel_split']['median'],
                    'caution':'Separate contexts can overlap. About 1.45x is the two-matrix microbenchmark gain vs serial split-core; about 1.10x is incremental vs same-core concurrent. This does not establish a 1.45x whole-model gain or fully attribute improvement to physical-core execution. Initial sequential-case A53 probe was unstable and is retained.'},
 'counterfactuals':{'profiled_seconds_if_scan_and_all_nonruntime_adapter_cost_zero':total-inclusive['scan']-(inclusive['linear.rknn']-native['rknn_matmul_run']),
                    'assumptions':'Only these two categories removed, same serial topology and all other phases unchanged. An idealized accounting exercise, not a forecast or hardware lower bound.'},
 'next_candidates':[
   {'priority':1,'change':'Fuse BF16->FP16 conversion with finite/overflow validation directly into reusable RKNN A buffers; convert C FP32->BF16 with identical rounding before leaving own bridge.',
    'evidence':'Finite validation ~74ms; input conversion ~26ms; adapter output/self ~63ms; explicit input/output copy ~35ms; allocation ~6ms.',
    'required_checks':'Preserve BF16 rounding, NaN/Inf and FP16-overflow errors, noncontiguous tensors, unsupported dtype/shape fallback and buffer lifetime. No disabled checks or result cache.'},
   {'priority':2,'change':'Schedule independent projections within one layer using public MatMul contexts; compare same-core concurrency, split-core concurrency, and concatenated projections.',
    'evidence':'Interleaved real-shape paired MatMul test 5.62ms serial ->3.89ms split concurrent; 30/30 rounds faster, exact output. Same-core concurrent already4.28ms.',
    'required_checks':'Actual D3 output parity across original seven cases, input reuse safety, model main-thread lifecycle and CPU thread contention. Layer-to-layer dependencies remain serial.'},
   {'priority':3,'change':'Specialize CPU GatedDeltaNet prefill: fuse normalization/scaling/padding and layout conversions, then optimize small batched FP32 matrix multiplies.',
    'evidence':'Six scans about248-255ms. Main operator hotspot is18 small batched matrix multiplies; six triangular solves are only about24-27ms.',
    'required_checks':'Keep causal scan, full attention visibility, zero initial state, long-input/state fallback, masking and dtype boundaries. Do not assume scan matmuls are independent of previous layers.'}],
 'two_hundred_ms_conclusion':'Another about4.3x is required from the 859ms deployed baseline. There is substantial work in own CPU/bridge code, but 200ms is unproven; current synchronous Runtime wall alone is about199ms. All major categories and NPU scheduling must improve together.',
 'service_preservation':service,
 'additional_latency_observation':{'source':'service-preservation.json','seconds':initial_service['inference_http_seconds'],
   'median':initial_service['inference_http_median'],'maximum':max(initial_service['inference_http_seconds']),
   'interpretation':'One first production request after an isolated test measured1.136s; retain it. Cause was not isolated. Previous warm100/100<1s does not imply idle/recovery/long-term hard latency guarantee.'},
 'verification_commands':['python3 -m py_compile bottleneck-analysis/*.py','bash -n bottleneck-analysis/run-profile.sh bottleneck-analysis/run-scan.sh',
                          'bash /opt/decision3-edge-bottlenecks-20261017/run-profile.sh','bash /opt/decision3-edge-bottlenecks-20261017/run-scan.sh',
                          'isolated probe_cores.py / probe_parallel.py / probe_parallel_interleaved.py containers','python3 bottleneck-analysis/verify_preservation.py'],
 'tests_reused':'Existing164 operator checks / seven-case model regression not rerun; no math/model optimization deployed. New tests cover instrumentation equality, six actual scan inputs, public core-mask behavior, concurrent contexts, and production preservation.'}
(root/'analysis-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Instrumented partition (ms):',{k:round(v*1000,2) for k,v in phases.items()})
print('Counterfactual ms:',result['counterfactuals']['profiled_seconds_if_scan_and_all_nonruntime_adapter_cost_zero']*1000)
print('Parallel improvement:',parallel['median_paired_serial_over_parallel_speedup'])
print('Service preserved:',service['container_id'],service['median'],service['max'])
