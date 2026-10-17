"""Final experimental evidence; no deployment or modification of live service."""
import datetime,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
def read(name):return json.loads((root/name).read_text())
r=read('logs/model-results.json');unit=read('logs/unit-results.json');shadow=read('logs/shadow-check.json')
pair=read('logs/paired-results.json');service=read('service-preservation-final.json')
assert all(x['passed'] for x in (r,unit,shadow,pair,service))
for filename,sha in r['source_sha256'].items():
    assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==sha,filename
variants={}
for name,v in r['variants'].items():
    d=max(x['vs_three_thread_baseline']['max_probability_or_score_delta'] for x in v['cases'].values())
    assert d<=.01 and all(x['vs_three_thread_baseline']['choices_unchanged'] for x in v['cases'].values())
    cpu_d=max((x['vs_historical_cpu']['max_probability_or_score_delta'] for x in v['cases'].values() if 'vs_historical_cpu' in x),default=0)
    assert cpu_d<=.01
    variants[name]={'count':v['timing']['count'],'median':v['timing']['median'],'p95':v['timing']['p95_nearest_rank'],
                    'max':v['timing']['max'],'model_cases':len(v['cases']),'max_delta_vs_starting_baseline':d,
                    'max_delta_vs_historical_cpu':cpu_d,'passed':v['passed'],'group_stats':v.get('group_stats')}
assert len(shadow['records'])==78
assert all(x['backend_before']['torch']==3 and x['backend_before']['openblas']==3 and x['backend_before']['omp_limit']==3 for x in pair['records'])
selected={k:v for k,v in r['selected_100'].items() if k not in ('seconds','response')}
result={'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'deployment_performed':False,
 'checkpoint':'vllm-sr/d3-edge@20c745653400d191ecacd99c4eafc0c5900dc15e',
 'constraints':{'weights_and_12_layers_unchanged':True,'noncausal_full_attention_and_causal_delta_scan_unchanged':True,
                'vendor_binaries_unmodified_no_reverse_engineering':True,'original_three_thread_service_preserved':True},
 'scope':'RK3576, CPU4-7, Torch/OpenMP3, loaded and warmed model, 115-token single question, concurrency1. In-process full system_one latency, excludes HTTP/network/startup; no response memoization.',
 'implementations':{'fusion':'BF16->FP16 plus finite/overflow check directly in RKNN A buffer; direct C FP32->BF16 RNE conversion into a new owned Torch output, retaining bias and unsupported-route behavior.',
                    'parallel':'24 scoped forks per text forward:12 MLP gate/up,6 attention Q/K/V,6 GatedDeltaNet qkv/z. Two persistent task queues, independent public RKNN contexts on masks1/2. All jobs join before reuse/error propagation.',
                    'combined':'Fusion + scoped parallel projection groups; convert common BF16 input once per fork, copy prepared FP16 inputs to independent handle buffers. No cross-request activation/result cache.'},
 'sequential_ablation':variants,
 'ablation_caution':'Initial baseline median855.7ms, later recheck806.7ms and a small output difference. Do not attribute all drift to candidate code. Only-parallel817.1ms lies inside this bracket, so standalone parallel gain is not established by this sequence.',
 'controlled_paired':{'summary':pair['summary'],'median_paired_speedup':pair['median_paired_speedup'],
                       'median_latency_reduction_fraction':1-pair['summary']['combined']['median']/pair['summary']['baseline']['median'],
                       'candidate_faster_rounds':pair['candidate_faster_rounds'],'rounds':10,
                       'max_timed_output_delta':max(x['max_delta_vs_baseline'] for x in pair['records']),
                       'english_three_question_delta':pair['prime']['combined']['english_delta'],
                       'scope':pair['scope'],
                       'caution':'Two full78-handle engines coexist with4096 test-container FD limit; configuration differs from the separate single-engine100-run test. Each trial resets Torch3 and has one explicit warmup. Not a deployment forecast.'},
 'selected_single_engine_100':selected,
 'precision':{'cases':7,'questions':14,'all_choices_unchanged':True,'max_probability_or_score_delta_vs_initial_three_thread_baseline':variants['combined']['max_delta_vs_starting_baseline'],
              'max_delta_vs_historical_cpu':variants['combined']['max_delta_vs_historical_cpu'],'cpu_reference_cases':5,
              'tolerance':.01,'confidence_and_legend_excluded':True,
              'shadow_78_real_projections_bitwise_equal':True,'controlled_paired_single_and_english_outputs_equal':True,
              'caution':'Seven-case regression and operator equality are not a full task-dataset accuracy evaluation. Whole-model sequential comparisons are not bitwise identical; observed differences remain below agreed tolerance.'},
 'focused_checks':unit,
 'iterations_and_failures':[
    {'record':'logs/unit-attempt01-failed.log','issue':'FP32 NaN -> BF16 canonical representation initially0xffff instead of Torch0x7fc0; finite values matched. Corrected both scalar/vector converters; exhaustive/random focused checks rerun and passed.'},
    {'record':'logs/shadow-fd-limit-failed.log','issue':'Double-engine shadow hit test-container default1024 open-file limit, SDK exited139. Repeated with isolated-container nofile4096 and core0; shadow/paired checks passed. Original service limits untouched. Generated3GB test core dump removed; error log retained.'}],
 'current_service':service,
 'reproduction':{'board_directory':'/opt/decision3-edge-bridge-parallel-20261017','existing_image':(root.parent/'optimized-deployment/image-id.txt').read_text().strip(),
                 'commands':['python3 -m py_compile bridge-parallel-experiment/*.py','bash -n bridge-parallel-experiment/run.sh bridge-parallel-experiment/run-shadow.sh bridge-parallel-experiment/run-paired.sh',
                             'bash /opt/decision3-edge-bridge-parallel-20261017/run.sh','bash /opt/decision3-edge-bridge-parallel-20261017/run-shadow.sh','bash /opt/decision3-edge-bridge-parallel-20261017/run-paired.sh','python3 bridge-parallel-experiment/verify_preservation.py']},
 'source_sha256':r['source_sha256'],
 'limitations':['No candidate HTTP service or production replacement.','200ms not reached.','100-request speed test compares its last response; full precision evidence is seven cases and all20 controlled paired timed outputs.','No new long-text/video/high-concurrency benchmarks; dynamic rows and CPU fallback checked at operator level.','Original CPU scan/norm/conv/attention kernels unchanged; existing164 arithmetic tests not duplicated.']}
(root/'experiment-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('Passed. Paired:',result['controlled_paired']['summary'],'speedup:',pair['median_paired_speedup'])
print('Selected100:',{k:v for k,v in selected.items() if k!='comparison'})
print('Maximum model delta:',result['precision']['max_probability_or_score_delta_vs_initial_three_thread_baseline'])
