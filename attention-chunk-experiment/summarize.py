import json,math,statistics,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
read=lambda p:json.loads((root/p).read_text())
paired=read('logs/qualified-paired.json');selected=paired['selected'];single=read('logs/final-single-'+selected+'.json')
full=read('logs/final-validate-combined.json');core=read('logs/final-validate-scan-core.json');screen=read('logs/variants-all.json');service=read('service-after.json')
assert selected=='scan' and paired['qualified_variants']==['baseline','scan'] and paired['passed'] and single['passed'] and service['passed']
assert len(single['records'])==100 and single['timing']['count']==100 and len(paired['records'])==40
assert set(r['variant'] for r in paired['records'])=={'baseline','scan'}
assert single['timed_handles']==72 and single['graph_contexts']==0 and single['counts_at_timing_end']['scan']=={'optimized':612,'fallback':0}
assert len(single['cases'])==7 and sum(len(v['response']['answers']) for v in single['cases'].values())==14
assert all(r['passed'] for r in single['records']) and all(v['passed'] for v in single['cases'].values())
values=single['timing']['seconds'];assert values==[r['seconds'] for r in single['records']]
assert single['timing']['p95_nearest_rank']==sorted(values)[math.ceil(.95*len(values))-1]
assert full['variants']['scan']['passed'] and not full['variants']['graph']['passed'] and not full['variants']['combined']['passed']
assert not core['variants']['core']['passed'] and not core['variants']['scan-core']['passed']
extra=full['variants']['scan']['extra'];assert len(extra)==12 and all(v['comparison']['passed'] for v in extra.values())
for name in ['baseline.py','chunk_blas2.py','chunk_blas2.c','libchunk_blas2.so','core_eval.py']:
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==single['sources'][name],name
units={p:read('logs/'+p)['passed'] for p in ['chunk-unit-replay.json','chunk-v2-unit-replay.json','chunk-blas-unit-replay.json','chunk-blas2-unit-replay.json','chunk-wrapper-unit.json','graph-guard-unit.json','core-guard-unit.json','graph-mask-unit.json','core-mask-unit.json']}
assert all(units.values())
def maxextra(row):return max(v['comparison']['max_delta'] for v in row['extra'].values())
neg={'whole_attention':maxextra(full['variants']['graph']),'whole_attention_plus_scan':maxextra(full['variants']['combined']),'attention_core':maxextra(core['variants']['core']),'attention_core_plus_scan':maxextra(core['variants']['scan-core'])}
num={'base_cases':7,'base_questions':14,'extra_shape115_cases':12,'extra_questions':12,'choices_unchanged':True,
     'main_max_delta_vs_454':max(v['vs_454']['max_delta'] for v in single['cases'].values()),'main_max_delta_vs_fp32':max(v['vs_fp32']['max_delta'] for v in single['cases'].values()),
     'extra_max_delta_vs_454':maxextra(full['variants']['scan']),'threshold':.01,'not_task_accuracy':True}
result={'selected':'CPU block32/value-tile64 scan with OpenBLAS SGEMM+STRSM','single100':single['timing'],'paired20':paired['timing'],'paired_summary':paired['paired']['scan'],'numerical':num,
        'rejected_probability_deltas':neg,'fixed_shape_graph_scope':'B1/T115 only; unsupported calls use frozen fallback','unit_tests':units,'production_unchanged':True,
        'verified_runtime_hashes':{name:single['sources'][name] for name in ['baseline.py','chunk_blas2.py','chunk_blas2.c','libchunk_blas2.so','core_eval.py']},
        'optional_profiling':'not available; two attempts crashed in an isolated profiling process after initialization; unresolved in own wrapper or SDK; no valid per-op runtime profile claimed',
        'metadata_note':'qualified-paired top-level variant inherited unused CLI default scan-core; actual qualified_variants/selected/every record are baseline/scan. Future header fixed without changing measured values.'}
(root/'experiment-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['完整注意力融合与块矩阵扫描：独立测试结果','',
       '本轮可保留的合格候选只有CPU块矩阵扫描，未达到200ms，收益较小。原权重/12层/非因果完整注意力/因果Delta语义保留，不反编译、不构建镜像、不部署。线上三线程服务未变。','',
       f"最终单候选100次：中位数 {single['timing']['median']*1000:.2f}ms，P95 {single['timing']['p95_nearest_rank']*1000:.2f}ms，最快 {single['timing']['min']*1000:.2f}ms，最慢 {single['timing']['max']*1000:.2f}ms。固定115-token单问题暖态system_one，不含HTTP；逐次校验，保留所有离群值，nearest-rank P95。",
       f"20轮交替同模型同72句柄对照：baseline {paired['timing']['baseline']['median']*1000:.2f}ms -> scan {paired['timing']['scan']['median']*1000:.2f}ms；逐轮改善中位数 {paired['paired']['scan']['round_reduction_median']*100:.2f}%，{paired['paired']['scan']['faster_rounds']}/20轮更快。每次计时前一次不计时预热；切回baseline后数值校验通过。",'',
       f"原7案例14问选择一致；相对454ms候选最大概率/score/noul差 {num['main_max_delta_vs_454']:.11g}，相对旧FP32参考 {num['main_max_delta_vs_fp32']:.11g}。另12个相同115-token形状的合成状态均保持选择，最大差 {num['extra_max_delta_vs_454']:.11g}。全部低于0.01。额外集只对照454ms基线，不伪称有其旧FP32参考。不是带标签任务准确率。",'',
       '有效改动：chunk_blas2.c/py，block32/value tile64；单位下三角求解和矩阵更新实现同一递推，FP32状态不变；外层OpenMP3，NumPy内OpenBLAS ILP64单线程。保留状态/packed/超长/类型/设备/梯度等不支持路径的原实现，输入不裁剪。',
       '维护成本：依赖固定镜像内NumPy捆绑OpenBLAS的文件路径和ILP64符号，升级镜像/NumPy需重测；目前只有约2%的成对收益，不应描述成明显跃升或成熟上线方案。','',
       '完整注意力方向：',
       '- 公开RKNN接口成功编译并在真机运行完整层：QKV投影、Q/K归一化、RoPE、GQA、非因果key mask、QK-softmax-V、gate、O投影；6层均验证未来token能影响第一个query。编译器显示SDPA融合。',
       '- 严格保留BF16中间舍入的ONNX在常量折叠时报No corresponding Numpy type for Tensor Type. bfloat16，未成功生成。RKNN config公共文档对本非量化路径仅声明float16。',
       '- 不改权重、改为FP16执行的完整层能够提速，但扩展概率检查不合格；不能因为choice一致而忽略概率门槛。',
       '- 注意力核心备选仅融合QK-softmax-V，其余投影/归一化/RoPE/gate及外围舍入保留；修正I/O后依然未通过概率门槛，也不采用。',
       '- 完整和核心图都只编译B1/T115，其他输入走原路径；没有把真实请求截短/填充到测试长度。不是所有请求均获得图加速。','',
       '16次初筛（不与正式100次混为同口径）：']
for n,row in screen['variants'].items():lines.append(f"{n}: {row['timing']['median']*1000:.2f}ms，原7案例回归={row['passed']}；尚未计入额外12案例的最终资格检查")
lines.extend(['', '额外12案例导致排除的最大概率/分数差：'])
for n,d in neg.items():lines.append(f'{n}: {d:.11g} > 0.01，排除')
lines.extend(['', 'CPU局部重放只作机制筛选（各六层中位数之和，不含同样的整模型准备）：'])
for p in ['chunk-unit-replay.json','chunk-v2-unit-replay.json','chunk-blas-unit-replay.json','chunk-blas2-unit-replay.json']:
    d=read('logs/'+p);s=d['sum_layer_medians_ms'];best=min((k for k in s if k!='baseline'),key=s.get);lines.append(f"{p}: baseline {s['baseline']:.2f}ms，最好 {best} {s[best]:.2f}ms；定向方程/因果/边界检查{d['unit_cases']}项")
lines.extend(['', '接口错误与诊断边界：',
              '核心图v1误把RKNN的优化NHWC属性当成调用者数组布局，导致大幅偏差；现在明确传NCHW。查询属性和修改前后误差在core-io-diagnostic.json，旧结果core-v1-invalid-layout.json标记为无效，不算性能/精度结论。',
              '两种图均完成6层x3种真实输入mask检查；仅最后key可见的核心输出与CPU参考逐元素一致，支持非因果语义。假对象测试仅覆盖控制流/清理，未代替真机数值回归。',
              '额外的公开PERF_DETAIL性能采集尝试崩溃，边界日志表明初始化已返回；尚未定位自身wrapper或SDK根因，未宣称获得逐算子运行时报告。该辅助步骤发生在正式100次和7案例完成之后，不影响已保存的计时/概率结果。首次qualified_suite因此退出139，已将可选故障诊断移出默认脚本；没有为纯脚本收尾变更重复跑模型。',
              'qualified-paired.json顶层variant继承了未使用的CLI默认scan-core；实际qualified_variants/selected/全部逐次标签均为baseline/scan。后续只修复元数据标签，原始数值保留。','',
              '线上三线程容器ID/启动时间/配置/源码/响应均核对未变，实验容器已退出；没有推送或更新原HTML报告。',
              '验证：Python全目录语法检查，逐个shell bash -n；各CPU方程/因果/边界/状态检查；图mask和回退/构造失败/错误清理检查；7案例14问+12匹配形状案例；合格候选20轮对照、100次与7案例复核；preserve_service.py after；summarize.py。',
              '详细命令和实际失败阶段见REPRODUCTION.txt，全部逐次数据与失败日志见logs/。200ms未实现；不建议用已超出概率阈值的更快融合版本替换现有服务。'])
(root/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:9]))
