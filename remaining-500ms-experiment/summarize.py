"""Validate saved evidence, then write the standalone result note and JSON."""
import json,math,statistics,hashlib
from pathlib import Path
root=Path(__file__).resolve().parent
read=lambda name:json.loads((root/name).read_text())
base=read('logs/profile.json');candidate=read('logs/profile-conversion-conv-prepare.json')
single=read('logs/final-single-conversion-conv-prepare.json');paired=read('logs/final-paired-conversion-conv-prepare.json')
assert base['passed'] and candidate['passed'] and single['passed'] and paired['passed']
runtime_files=['loader.py','variants.py','conv_btc.py','conv_btc.c','prepared_scan.py','scan_prepare.c','conversion_bridge.cpp','libconv_btc.so','libscan_prepare.so','libconversion_bridge.so','final_compare.py']
for name in runtime_files:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==single['sources'][name],name
for name in ['diagnose_candidate.py','profile_conversion.cpp','libprofile_conversion.so']:
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==candidate['sources']['/work/'+name],name
assert len(single['records'])==100 and single['timing']['count']==100 and len(single['cases'])==7
assert sum(len(c['response']['answers']) for c in single['cases'].values())==14
assert len(paired['records'])==40 and paired['faster_rounds']>0
assert all(r['comparison']['passed'] and r['comparison_vs_fp32']['passed'] for r in single['records'])
assert all(c['comparison']['passed'] and c['comparison_vs_fp32']['passed'] for c in single['cases'].values())
assert all(c['optimized']==654 and c['fallback']==0 for c in single['cpu_counts'].values())
assert single['handles']['candidate']==72 and single['engine_stats']['packed_range_fallbacks']==0
assert single['handles']['baseline']==0
values=single['timing']['seconds'];assert values==[r['seconds'] for r in single['records']]
assert single['timing']['p95_nearest_rank']==sorted(values)[math.ceil(.95*len(values))-1]

def partition(report):
    rows=[]
    for record in report['records']:
        spans={k:v['exclusive'] for k,v in record['spans'].items()}
        assert abs(record['total']-sum(spans.values()))<1e-7
        assert len(record['native_handles'])==72 and all(h['calls']==1 for h in record['native_handles'])
        bridge=sum(v for k,v in spans.items() if k.startswith('bridge.') or k=='linear.rknn')
        row={'CPU扫描及前处理':spans['scan'],'SDK同步区间并集':record['sdk_union'],'桥接其余开销':bridge-record['sdk_union'],
             'SiLU':spans['silu'],'归一化':spans['norm'],'CPU完整注意力SDPA':spans['sdpa'],'因果卷积（不含SiLU）':spans['causal_conv'],'RoPE':spans['rotary']}
        row['其他CPU/准备/调度']=record['total']-sum(row.values());rows.append(row)
    result={k:statistics.mean(r[k] for r in rows)*1000 for k in rows[0]}
    result['插桩总计']=statistics.mean(r['total'] for r in report['records'])*1000
    return result
bpart,cpart=partition(base),partition(candidate)
units={name:read('logs/'+name)['passed'] for name in ['conv-unit-replay.json','prepare-unit.json','conversion-unit.json','conversion-packing-unit.json']};assert all(units.values())
service=read('service-after.json');assert service['passed'] and not service['remaining_test_containers']
case_delta=max(x['comparison']['max_delta'] for x in single['cases'].values());fp32delta=max(x['comparison_vs_fp32']['max_delta'] for x in single['cases'].values())
screens={f:read('logs/'+f)['variants'] for f in ['screen.json','screen2.json']}
result={'scope':'RK3576独立进程；115-token单问题暖态system_one，并发1，不含HTTP；原权重12层与注意力语义保留；未部署、未推送。',
        'selected_variant':single['variant'],'single100':single['timing'],'pair20':{'baseline':paired['timing']['baseline'],'candidate':paired['timing']['candidate'],'median_round_reduction':paired['paired_reduction_median'],'faster_rounds':paired['faster_rounds'],'scope':paired['scope']},
        'numerical':{'cases':7,'questions':14,'choices_unchanged':True,'max_delta_vs_latest':case_delta,'max_delta_vs_fp32':fp32delta,'threshold':.01,'not_task_accuracy':True},
        'profile_mean_ms':{'baseline':bpart,'candidate':cpart},'unit_tests':units,'production_unchanged':True,
        'verified_runtime_source_hashes':{name:single['sources'][name] for name in runtime_files},'tested_directory_snapshot':single['sources'],'untimed_postprocessor':'summarize.py (may change after timings)'}
(root/'experiment-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['500ms余量剖析及独立优化结果','',result['scope'],'',
       f"最终候选单引擎100次：中位数 {single['timing']['median']*1000:.2f}ms，P95 {single['timing']['p95_nearest_rank']*1000:.2f}ms，最快 {single['timing']['min']*1000:.2f}ms，最慢 {single['timing']['max']*1000:.2f}ms。保留所有逐次数据，每次输出均校验。",
       f"20轮交替对照：基线 {paired['timing']['baseline']['median']*1000:.2f}ms -> 候选 {paired['timing']['candidate']['median']*1000:.2f}ms；逐轮时延降低中位数 {paired['paired_reduction_median']*100:.2f}%；{paired['faster_rounds']}/20轮更快。两持久引擎，与单引擎100次资源条件不同。",
       f"7案例14问题选择一致；相对约510ms候选最大概率/score/noul差 {case_delta:.11g}，相对旧FP32输出参考 {fp32delta:.11g}，均<0.01。不是带标签任务准确率。",'',
       '真正有效的改动：',
       '1. 原卷积前把BTC数据转为BCT的复制很贵，改为C内核直接读取BTC并返回原BCT视图；局部真实输入重放约55.47 -> 27.31ms，局部输出逐元素一致。',
       '2. 扫描前BF16转FP32、布局调整、Q/K归一化和query缩放合并为一次遍历；调用原来的同一递推扫描内核，不改变因果逻辑。FP32归约次序有小差异。',
       '3. 桥接格式转换对常见正常值改用NEON整数位运算，特殊值回原浮点路径；所有16位模式验证非NaN位一致，保护状态一致。没有新增量化/更改已有浮点边界。','',
       '分段剖析：下表是各10次有插桩均值（ms），其总计含测量开销，不能硬凑成500ms或最终100次中位数。SDK为调用同步区间并集，包含提交/驱动等待，并非纯设备计算；worker重叠未相加。']
for k in bpart:lines.append(f'{k}: {bpart[k]:.2f} -> {cpart[k]:.2f}')
lines.extend(['',f"无Python插桩前/后中位数：原候选 {base['before']['median']*1000:.2f} / {base['after']['median']*1000:.2f}ms；新候选 {candidate['before']['median']*1000:.2f} / {candidate['after']['median']*1000:.2f}ms。C时钟插桩仍存在；正式成绩取单引擎100次。",'', '筛选结果（每种16次，中位数；资源/运行先后不同，收益最终由成对试验复核）：'])
for file,variants in screens.items():
    for name,row in variants.items():
        lines.append(f"{file}/{name}: {row['timing']['median']*1000:.2f}ms，回归通过={row['passed']}")
replays=read('logs/replay-scan.json')
lines.append('两种扫描代数重排实现为负结果（真实输入局部重放、6层各15轮，包含准备，不是整模型成绩）：')
for name in replays[0]['times']:
    lines.append(f"{name}: 各层中位数合计 {sum(statistics.median(r['times'][name]) for r in replays)*1000:.2f}ms")
lines.extend(['', '未采用：显式cacheable内存+手工同步没有收益，两种扫描代数重排也更慢；未部署负结果。',
              '距离200ms仍大：剩余NPU同步、CPU扫描、完整注意力和桥接/激活/归一化分散在多个阶段。下一轮应优先验证连续注意力子图的公开RKNN实现以减少提交与布局边界，以及小块矩阵化的CPU扫描以减少逐token状态读写；这些是未验证方向，不承诺收益。已试过的整块CPU编译融合无收益，不建议原样重试。',
              '仍需真实业务回归与端到端HTTP/并发验证；本轮不是上线版本。模型、原权重/12层和mask未变，不反编译。',
              '线上三线程容器ID/启动时间/配置/源文件/响应保持，实验容器已移除。',
              '验证：python3 -m py_compile remaining-500ms-experiment/*.py；逐个bash -n检查remaining-500ms-experiment/*.sh；test_conv.py；test_prepare.py；test_conversion.py；test_bridge.py；screen.py；screen2.py；final_suite.sh（20轮对照、100次、7案例与分段）；preserve_service.py after；summarize.py。',
              '详细复现命令见REPRODUCTION.txt，逐次数据见logs/，源码与本轮实际测试文件hash见对应JSON。没有自动部署、构建镜像、提交或推送。'])
(root/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:7]))
