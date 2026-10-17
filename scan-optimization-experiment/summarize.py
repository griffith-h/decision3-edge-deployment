"""Check measured artifacts and write a compact, inspectable final result."""
import hashlib,json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
full=json.loads((root/'logs/full-results.json').read_text())
replay=json.loads((root/'logs/replay-results.json').read_text())
service=json.loads((root/'service-after.json').read_text())
assert full['passed'] and service['passed']
for name in ('scan.py','scan_native.c','benchmark.py','test_scan.py'):
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==full['sources']['/work/'+name],name
for name in ('scan.py','scan_native.c'):
    assert replay['sources'][name]==full['sources']['/work/'+name],name
for name in ('adapter.py','fused_bridge.cpp'):
    assert hashlib.sha256((root.parent/'bridge-parallel-experiment'/name).read_bytes()).hexdigest()==full['sources']['/combined/'+name],name
assert full['units']['passed'] and len(full['units']['equation_checks'])==480
selected=full['selected_100']['variant'];assert full['model'][selected]['passed']
assert full['selected_100']['every_response_checked'] and full['selected_100']['count']==100
records=full['paired']['records'];summary=full['paired']['summary']
comparisons={}
for name,record in full['model'].items():
    paired=[]
    for i in range(12):
        baseline=next(r['seconds'] for r in records if r['round']==i and r['variant']=='baseline')
        matches=[r for r in records if r['round']==i and r['variant']==name]
        if matches:paired.append(matches[0]['seconds']/baseline)
    comparisons[name]={'passed':record['passed'],
        'max_delta_vs_same_process_combined':max(c['comparison']['max_probability_or_score_delta'] for c in record['cases'].values()),
        'choices_unchanged':all(c['comparison']['choices_unchanged'] for c in record['cases'].values()),
        'case_scan_calls':{k:v['scan_calls'] for k,v in record['cases'].items()}}
    if paired:
        comparisons[name].update(paired_median=summary[name]['median'],median_paired_latency_reduction=statistics.median(1-v for v in paired),
                                 faster_rounds=sum(v<1 for v in paired),paired_rounds=len(paired))
budget=full['budget']['medians']
result={'passed':True,'scope':full['scope'],'case_count':full['case_count'],'question_count':full['question_count'],
        'settings':full['settings'],'selected':selected,'paired_summary':summary,'comparisons':comparisons,
        'selected_100':full['selected_100'],'budget':budget,
        'measured_remaining_after_zero_scan':budget[selected]['other'],
        'target_seconds':.2,'target_reached':full['selected_100']['median']<=.2,
        'source_hashes_verified':True,'native_equation_checks':len(full['units']['equation_checks']),
        'exploratory_real_input_replay':{'scope':'Sequential microbenchmark of six captured 115-token layer inputs; not paired model speedup.',
            'baseline_sum_medians':replay['variants']['baseline']['sum_medians'],
            'selected_sum_medians':replay['variants'][selected]['sum_medians'],
            'selected_max_scan_output_abs':max(r['max_abs'] for r in replay['variants'][selected]['layers'])},
        'max_unit_relative_l2_vs_official_recurrent':max(r['relative_l2_vs_official_recurrent'] for r in full['units']['equation_checks']),
        'native_supported_tokens':'1..512, all other state/type/shape/packed contracts delegate unchanged',
        'production_preserved':True,'production_id':service['container_id'],'production_started_at':service['started_at'],
        'production_after_http_seconds':service['http_seconds'],'production_after_http_median':service['median'],
        'limitations':['115-token single-question warmed model call, concurrency 1; not HTTP or cold startup.',
                      'Seven cases are regression checks, not task accuracy on a labeled evaluation set.',
                      'Other-time residual is a current-path budget, not a hardware/theoretical latency lower bound.',
                      'Exploratory replay variants run sequentially; primary speed evidence is same-engine alternating model trials.',
                      'No production cutover or image build. Early fallback-only and layout-failing runs retained as failed diagnostics.']}
(root/'experiment-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['CPU GatedDeltaNet 扫描优化结果（独立实验，未部署）','',
       f"组合基线：{summary['baseline']['median']*1000:.3f}ms（12轮交替整模型计时中位数）"]
for name,c in comparisons.items():
    if 'paired_median' in c:lines.append(f"{name}: {c['paired_median']*1000:.3f}ms；逐轮降时延中位数{c['median_paired_latency_reduction']:.2%}；{c['faster_rounds']}/{c['paired_rounds']}轮更快；7用例最大概率/分数差{c['max_delta_vs_same_process_combined']:.10g}；选择一致={c['choices_unchanged']}")
st=full['selected_100']
lines.extend(['',f"选中{selected}，连续100次：中位数{st['median']*1000:.3f}ms / P95 {st['p95_nearest_rank']*1000:.3f}ms / 最慢{st['max']*1000:.3f}ms；每次输出均校验，最大差{st['max_delta']:.10g}。",
    f"扫描轻量计时：基线{budget['baseline']['scan']*1000:.3f}ms -> 选中{budget[selected]['scan']*1000:.3f}ms；其余部分残差中位数{budget[selected]['other']*1000:.3f}ms。",
    f"200ms目标尚未达到。把本次候选扫描开销假设为零，其余当前路径仍约{budget[selected]['other']*1000:.0f}ms；这不是硬件理论下界。",
    f"原始模型、12层和注意力语义保留；480项递推/边界检查、{full['case_count']}用例{full['question_count']}问题回归通过。概率差不等于准确率下降；没有带标签的数据集质量结论。",
    f"线上三线程容器ID/启动时间/配置/源码/示例响应均保持；结束后5次HTTP中位数{service['median']*1000:.3f}ms。实验容器已移除。",
    '', '时延口径：预热后115token、单问题、并发1，完整system_one，不含加载/HTTP/网络。',
    '正式对照只切换扫描函数，同一模型/同一个78句柄组合引擎；按轮交替顺序、每组预热一次，避免把跨进程漂移算作提速。',
    '原生路径1..512token；更长序列、初始/返回状态、packed输入或不支持类型/形状走原扫描，输入不被裁剪。',
    '源码：scan.py + scan_native.c；复现：REPRODUCTION.txt + run.sh；原始逐次数据：logs/full-results.json。',
    '早期两轮资格回退不能算优化成绩；FP32 decay 非连续布局问题已修复，失败日志保留。',
    '验证命令：python3 -m py_compile scan-optimization-experiment/*.py；bash -n scan-optimization-experiment/run.sh；run.sh --stage full；preserve_service.py after；summarize.py。'])
(root/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:8]))
