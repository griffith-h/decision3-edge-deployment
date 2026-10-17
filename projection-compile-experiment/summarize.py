"""Aggregate inspectable experimental evidence, keeping timing scopes separate."""
import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent;logs=root/'logs'
def read(name):return json.loads((logs/name).read_text())
rows=[]
for p in sorted(logs.glob('*.json')):
    d=json.loads(p.read_text())
    if 'timing' not in d:continue
    cs=d.get('cases',{});t=d['timing']
    rows.append({'file':p.name,'tag':d.get('tag'),'backend':d.get('packed_backend','tensor-or-baseline'),
        'count':t['count'],'median_ms':1000*t['median'],'p95_ms':1000*t['p95_nearest_rank'],'min_ms':1000*t['min'],'max_ms':1000*t['max'],
        'max_case_delta_vs_safe':max((c['comparison']['max_delta'] for c in cs.values()),default=None),
        'max_case_delta_vs_fp32':max((c['comparison_vs_fp32']['max_delta'] for c in cs.values()),default=None),
        'passed':d.get('passed',False),'case_count':len(cs),'handles':d.get('handles'),
        'cold_first_call_seconds':d.get('cold_first_call_seconds'),'compile_stats':d.get('compile_stats'),
        'engine_stats':d.get('engine_stats')})
pairs=[]
for p in sorted(logs.glob('*paired*.json')):
    d=json.loads(p.read_text())
    if 'summary' not in d:continue
    baseline=d['summary']['baseline']['median'];candidate=d['summary']['candidate']['median']
    records=d['records'];rounds=sorted({r['round'] for r in records});reductions=[]
    for i in rounds:
        trial={r['variant']:r['seconds'] for r in records if r['round']==i}
        reductions.append(1-trial['candidate']/trial['baseline'])
    pairs.append({'file':p.name,'baseline_ms':baseline*1000,'candidate_ms':candidate*1000,
                  'median_per_round_reduction_percent':100*statistics.median(reductions),'faster_rounds':sum(v>0 for v in reductions),'rounds':len(rounds)})
units={p.name:json.loads(p.read_text()).get('passed',False) for p in logs.glob('*unit.json')}
after=json.loads((root/'service-after.json').read_text()) if (root/'service-after.json').exists() else {}
result={'scope':'RK3576；原权重、12层、非因果完整注意力；同一115-token请求的暖态system_one耗时，不含HTTP。7用例14问题为数值回归，不能代表任务准确率。无反编译、镜像构建、主机时钟或线上改动。',
        'screening_and_stability':rows,'paired':pairs,'units':units,'production_preserved':after.get('passed',False)}
(root/'experiment-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
lines=['同输入投影合并、两核平衡、CPU编译融合：独立测试结果','',result['scope']]
chosen_path=logs/'scatter-balanced-none.json'
if chosen_path.exists():
    chosen=json.loads(chosen_path.read_text());t=chosen['timing']
    if t['count']==100:
        assert chosen['passed'] and chosen['every_response_checked'] and chosen['handles']==72
        assert chosen['case_count']==7 and chosen['question_count']==14
        best_pair=next(v for v in pairs if v['file']=='scatter-paired-balanced-none.json')
        max_delta=max(v['comparison']['max_delta'] for v in chosen['cases'].values())
        max_fp32=max(v['comparison_vs_fp32']['max_delta'] for v in chosen['cases'].values())
        result['selected']={'file':chosen_path.name,'median_ms':1000*t['median'],'p95_ms':1000*t['p95_nearest_rank'],
                            'count':100,'max_delta_vs_safe':max_delta,'max_delta_vs_fp32':max_fp32,'paired':best_pair,'deployed':False}
        lines+=['',f"本轮最好候选：合并同输入投影＋两核等宽分块＋原地转换写回；保留原CPU路径及输出范围回退。100次中位数 {1000*t['median']:.2f}ms，P95 {1000*t['p95_nearest_rank']:.2f}ms，最慢 {1000*t['max']:.2f}ms。未到200ms。",
                f"成对复核：{best_pair['baseline_ms']:.2f} -> {best_pair['candidate_ms']:.2f}ms，逐轮降时延中位数 {best_pair['median_per_round_reduction_percent']:.2f}%，{best_pair['faster_rounds']}/{best_pair['rounds']}轮更快；与单引擎100次条件不同。",
                f"7用例14问题选择一致；相对512ms安全基线，受检概率/score/noul最大绝对差 {max_delta:.9g}；相对更早FP32输出基线为 {max_fp32:.9g}。100次逐次校验通过。",
                'CPU编译：动态MLP、大区域动态、大区域固定形状、编译内核单线程及组合均无端到端收益，不纳入候选。固定形状只优化暖机签名，其他签名回退原路径。',
                '首个CPU编译方案丢失中间BF16舍入边界，单测失败后用显式位运算修复；没有放宽原阈值。首次调用含编译/NPU初始化，不计入暖态中位数，详见原始JSON。',
                '原线上三线程服务保留；候选未部署。']
        (root/'experiment-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
if (logs/'group-probe.json').exists():
    probe=read('group-probe.json');lines+=['','真实输入分组重放（包含SDK等待/输出转换，不是纯NPU计算；插桩，仅作机制诊断）：']
    for kind,g in probe['groups'].items():
        lines.append(kind+': '+', '.join(f"{n}={v['median']*1000:.3f}ms" for n,v in g['summary'].items()))
lines+=['','单引擎数据（时间ms；不同进程屏蔽不了漂移，最终以成对试验复核）：']
for r in rows:lines.append(f"{r['file']}: n={r['count']}, median={r['median_ms']:.2f}, P95={r['p95_ms']:.2f}, min={r['min_ms']:.2f}, max={r['max_ms']:.2f}, passed={r['passed']}, maxΔsafe={r['max_case_delta_vs_safe']}, maxΔFP32={r['max_case_delta_vs_fp32']}")
lines.append('\n同模型、持久引擎、交替顺序的成对复核（资源条件不同，不能与单引擎100次当同口径）：')
for r in pairs:lines.append(f"{r['file']}: baseline={r['baseline_ms']:.2f} -> candidate={r['candidate_ms']:.2f}, 逐轮降时延中位数={r['median_per_round_reduction_percent']:.2f}%, {r['faster_rounds']}/{r['rounds']}轮更快")
lines+=['','定向测试：'+str(units),'线上容器/源码/配置/响应不变：'+str(after.get('passed',False)),
        '验证命令：bash build_scatter.sh；bash run.sh scatter_runner.py test_packing.py；bash run.sh --compiler cpu_suite.sh；bash run.sh --compiler scatter_suite.sh（诊断脚本分支修复后继续cpp1_suite.sh）；bash run.sh --compiler cpp1_suite.sh；bash run.sh final_native_suite.sh；python3 preserve_service.py after；python3 summarize.py。',
        '本地静态检查：python3 -m py_compile projection-compile-experiment/*.py；bash -n projection-compile-experiment/*.sh。',
        '定向覆盖：分列/偏置/非连续输入/动态行数、输出溢出CPU回退与恢复、工作线程错误后的全部等待；CPU显式舍入100003位模式、eager对照、空输入/FP32/梯度回退、固定形状失配回退、RoPE及还原。',
        '所有逐次原始值、完整范围、首次签名耗时和源码/原生库哈希见logs/；完整复现见REPRODUCTION.txt。']
(root/'RESULTS.txt').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
