"""Evidence-based aggregate; keep instrumentation and speed trials distinct."""
import hashlib,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
def read(name):return json.loads((root/'logs'/name).read_text())
model=read('model-results.json');before=read('profile.json');half=read('profile-half.json')
envs={n:read('env-'+n+'.json') for n in ('default-before','passive','default-after')}
safe=read('env-safe-default.json');safe_pair=read('safe-paired.json');output_range=read('output-range-unit.json')
service=json.loads((root/'service-after.json').read_text())
assert service['passed'] and model['half_output']['passed'] and all(j['passed'] for j in envs.values())
assert safe['passed'] and safe_pair['passed'] and output_range['passed']
def partition(profile):
    groups=[]
    for r in profile['records']:
        spans=r['spans'];bridge=sum(v['exclusive'] for k,v in spans.items() if k.startswith('bridge.') or k=='linear.rknn')
        scan=spans['scan']['exclusive'];cpu=r['total']-bridge-scan
        row={'scan':scan,'sdk_sync_union':r['sdk_union'],'bridge_other':bridge-r['sdk_union']}
        row.update({k:spans[k]['exclusive'] for k in ('silu','norm','sdpa','causal_conv','rotary')})
        row['other_cpu']=cpu-sum(row[k] for k in ('silu','norm','sdpa','causal_conv','rotary'))
        assert abs(sum(row.values())-r['total'])<1e-7 and min(row.values())>=0
        groups.append(row)
    return {k:statistics.mean(r[k] for r in groups) for k in groups[0]}
paired=model['half_output']['paired'];ratios=[]
for i in range(12):
    b=next(r['seconds'] for r in paired['records'] if r['round']==i and r['variant']=='fp32')
    c=next(r['seconds'] for r in paired['records'] if r['round']==i and r['variant']=='fp16')
    ratios.append(c/b)
for j in envs.values():
    for f in ('half_bridge.cpp',):
        assert hashlib.sha256((root/f).read_bytes()).hexdigest()==j['sources']['/work/'+f],f
    assert hashlib.sha256((root/'logs/environment-benchmark-v1.py').read_bytes()).hexdigest()==j['sources']['/work/environment_benchmark.py']
    assert j['settings']['torch_threads']==3 and j['settings']['affinity']==[4,5,6,7]
candidate='safe-default';selected=safe
for f in ('safe_half_bridge.cpp','safe_engine.py','environment_benchmark.py','common.py'):
    assert hashlib.sha256((root/f).read_bytes()).hexdigest()==safe['sources']['/work/'+f],f
safe_ratios=[]
for i in range(6):
    b=next(r['seconds'] for r in safe_pair['records'] if r['round']==i and r['variant']=='fp32')
    c=next(r['seconds'] for r in safe_pair['records'] if r['round']==i and r['variant']=='safe_fp16')
    safe_ratios.append(c/b)
result={'passed':True,'constraints':'Original official model/12 layers/weights/attention semantics, independent process, no reverse engineering or image build, deployed three-thread service preserved.',
        'profile_uninstrumented':{'before':before['before'],'after':before['after']},
        'initial_profile_mean_partition':partition(before),'half_profile_mean_partition':partition(half),
        'instrumentation_note':'Exclusive means close to the instrumented request mean. SDK union is overlap-adjusted synchronous API wall time, not pure NPU compute. FP16 profile precedes the added range guard; the guarded candidate is separately timed and checked. Do not force these numbers to equal the previous uninstrumented 469ms residual.',
        'cpu_ablation':{n:{k:v for k,v in j.items() if k!='seconds'} for n,j in model['cpu_paired']['summary'].items()},
        'cpu_all_accuracy_max_delta':max(r['comparison']['max_delta'] for r in model['cpu']['all']['cases'].values()),
        'half_output_paired':paired['summary'],'median_paired_latency_reduction':statistics.median(1-r for r in ratios),
        'half_output_faster_rounds':sum(r<1 for r in ratios),
        'half_output_max_delta':max(r['comparison']['max_delta'] for r in model['half_output']['cases'].values()),
        'guarded_output_paired':safe_pair['summary'],'guarded_median_paired_latency_reduction':statistics.median(1-r for r in safe_ratios),
        'guarded_faster_rounds':sum(r<1 for r in safe_ratios),'guarded_range_tests':output_range,
        'environment_trials':{n:{'timing':j['timing'],'cpu_usage':j['cpu_usage'],'environment':j['environment']} for n,j in envs.items()},
        'selected_environment':candidate,'selected_timing':selected['timing'],
        'selected_max_case_delta':max(r['comparison']['max_delta'] for r in selected['cases'].values()),
        'selected_all_choices_unchanged':all(r['comparison']['choices_unchanged'] for r in selected['cases'].values()),
        'selected_every_timed_response_checked':selected['every_response_checked'],
        'selected_7_cases_checked':len(selected['cases'])==7,
        'selected_engine_stats':selected['engine_stats'],
        'target_seconds':.2,'target_reached':selected['timing']['median']<=.2,
        'production_preservation':{'passed':service['passed'],'container_id':service['container_id'],'started_at':service['started_at'],'http_median':service['median']},
        'limitations':['Warmed115-token single-question model latency, concurrency1; not HTTP/cold-start/high-concurrency SLO.',
                       'FP16 output changes numerical representation; public API does not prove identical internal accumulation.',
                       'Seven cases/14 questions are regression checks, not a labeled task accuracy evaluation.',
                       'Environment trials are ordered before/passive/after single-engine processes, not same-process paired evidence; small differences may be drift.',
                       'CPU math variants passed numeric checks but did not deliver meaningful end-to-end gains; not selected.']}
(root/'experiment-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['469ms余量分析与独立优化测试结果','',
       f"原候选无插桩复测中位数：{before['before']['median']*1000:.2f} / {before['after']['median']*1000:.2f}ms。",
       '以下是有插桩均值分解，含测量开销；不能硬凑为历史无插桩469ms。FP16参考剖析在输出范围保护加入前；最终带保护版的整体速度和精度另测。']
labels={'scan':'扫描','sdk_sync_union':'SDK同步区间并集','bridge_other':'桥接转换/分配/提交/等待余量','silu':'SiLU','norm':'Norm','sdpa':'CPU SDPA','causal_conv':'因果卷积','rotary':'RoPE','other_cpu':'其他CPU/准备/调度'}
for k,title in labels.items():lines.append(f"{title}: 原FP32输出 {result['initial_profile_mean_partition'][k]*1000:.2f}ms -> FP16输出 {result['half_profile_mean_partition'][k]*1000:.2f}ms")
lines.extend(['',f"公开SDK输出类型成对对照：FP32 {paired['summary']['fp32']['median']*1000:.2f}ms -> FP16 {paired['summary']['fp16']['median']*1000:.2f}ms；逐轮降时延中位数{result['median_paired_latency_reduction']:.2%}；{result['half_output_faster_rounds']}/12轮更快。",
    f"FP16 7用例14问题选择一致，最大概率/score/noul绝对差{result['half_output_max_delta']:.9g}；原始权重/层数/注意力语义保留，没有量化权重。",
    'SDK只公开输入输出类型，不能据此断言内部累加完全相同；不称无损。',
    f"最终带保护版本严格对照：FP32 {safe_pair['summary']['fp32']['median']*1000:.2f}ms -> safe FP16 {safe_pair['summary']['safe_fp16']['median']*1000:.2f}ms，逐轮降时延中位数{result['guarded_median_paired_latency_reduction']:.2%}，{result['guarded_faster_rounds']}/6轮更快。",
    '输出范围测试：未保护FP16把原CPU可表示的1048576变为Inf；最终版本在转换时检测饱和/非有限值，回退该投影到原CPU forward；混合并行组、后续恢复和原输入校验均通过。',
    'CPU激活/归一化融合为负结果：'+', '.join(f"{n}={j['median']*1000:.2f}ms" for n,j in model['cpu_paired']['summary'].items()),
    '真实SiLU重放：Torch BF16优于Torch FP32转回和自写向量多项式；公开SLEEF函数的合成筛选也无收益，未进行整模型推广。',''])
for n,j in envs.items():lines.append(f"进程策略{n}: n={j['timing']['count']}, median={j['timing']['median']*1000:.2f}ms, P95={j['timing']['p95_nearest_rank']*1000:.2f}ms, max={j['timing']['max']*1000:.2f}ms, CPU秒/请求={j['cpu_usage']['cpu_seconds_per_request']:.3f}")
lines.extend(['',f"当前实验候选：FP16输出，环境{candidate}；中位数{selected['timing']['median']*1000:.2f}ms，P95 {selected['timing']['p95_nearest_rank']*1000:.2f}ms，最慢{selected['timing']['max']*1000:.2f}ms。",
    f"该版本7用例最大差{result['selected_max_case_delta']:.9g}；逐次输出已校验。实际模型中新输出范围回退次数={selected['engine_stats'].get('half_output_cpu_fallbacks',0)}。环境策略小差异按漂移审慎解释，不与SDK成对提速混为一谈。",
    '200ms尚未达到。FP16剖析中SDK同步和扫描两项合计仍约'+f"{(result['half_profile_mean_partition']['sdk_sync_union']+result['half_profile_mean_partition']['scan'])*1000:.0f}ms"+'，还未计其余CPU和桥接。此为当前执行路径，并非硬件理论下界。',
    '下一步的优先方向：用开源CPU编译器验证较大连续区域的融合；或用公开RKNN图/矩阵接口合并同输入投影并平衡两核负载，减少提交和数据边界。是否有效仍需实测，不能承诺200ms。',
    '线上三线程容器ID/启动时间/配置/源码/示例响应未变，所有临时实验容器已移除。',
    '验证：run.sh默认剖析；D3_REMAINDER_SCRIPT=benchmark.py run.sh；run.sh --half-output --probe-activations；D3_REMAINDER_SCRIPT=environment_benchmark.py run.sh；preserve_service.py after；summarize.py。',
    '详细命令见REPRODUCTION.txt；原始逐次结果见logs/。不构建镜像，不启动测试HTTP服务，不自动部署/提交/推送。'])
(root/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
