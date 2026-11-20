"""Build an aggregate-only comparison; never embed protected benchmark questions."""
import argparse
import collections
import hashlib
import html
import json
from pathlib import Path
import statistics

from decision_index.suite.io import read_jsonl
from score_sample import latency
from stages import STAGES


def results(folder):
    path = folder / 'results.jsonl'
    return {r['run_id']: r for r in read_jsonl(path, complete_lines_only=True)} if path.exists() else {}


def build(root, manifest, output):
    ids = set(manifest['run_ids_in_order'])
    protocol_path = root / 'protocol.json'
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    requested = protocol.get('requested_stages', [s[0] for s in STAGES])
    if not requested or set(requested) - {s[0] for s in STAGES}:
        raise ValueError('Unknown or empty requested evaluation scope')
    status = json.loads((root / 'run-status.json').read_text()) if (root / 'run-status.json').exists() else {}
    base = results(root / 'cpu-fp32')
    rows = []
    for stage, label, *_ in STAGES:
        if stage not in requested:
            continue
        raw = results(root / stage)
        if set(raw) - ids:
            raise ValueError('Foreign request found in stage ' + stage)
        counts = dict(collections.Counter(r['status'] for r in raw.values()))
        row = {'stage': stage, 'label': label, 'completed_requests': len(raw),
               'expected_requests': len(ids), 'counts': counts,
               'sample_complete': set(raw) == ids,
               'execution_status': status.get(stage, {}).get('status', 'pending')}
        summary = root / stage / 'sample-summary.json'
        if row['sample_complete'] and summary.exists():
            saved = json.loads(summary.read_text())
            for key in ('board_method', 'labelled_field_accuracy', 'sampled_public_index',
                        'method_by_input_tokens', 'native_benchmark_metrics', 'acceleration_counter_hits'):
                row[key] = saved[key]
            common = sorted(k for k in ids if base.get(k, {}).get('status') == 'ok'
                            and raw.get(k, {}).get('status') == 'ok')
            if len(base) == len(ids) and common:
                baseline = [base[k]['raw_output']['board_method_ms'] for k in common]
                current = [raw[k]['raw_output']['board_method_ms'] for k in common]
                row['paired_success_vs_cpu_fp32'] = {
                    'requests': len(common), 'cpu': latency(baseline), 'stage': latency(current),
                    'median_per_request_speedup': statistics.median(a / b for a, b in zip(baseline, current)),
                    'sum_latency_speedup': sum(baseline) / sum(current)}
        rows.append(row)
    requested_complete = all(r['sample_complete'] and 'board_method' in r for r in rows)
    all_complete = requested_complete and len(rows) == len(STAGES)
    final_complete = any(r['stage'] == 'native-bf16' and r['sample_complete'] and 'board_method' in r for r in rows)
    final_only = requested == ['native-bf16']
    for row in rows:
        # Per-group hit counts stay in private scoring files, not the public page.
        row.get('labelled_field_accuracy', {}).pop('groups', None)
    report = {'scope': 'same frozen complete-group stratified public-text sample; not full official leaderboard',
              'model_revision': protocol.get('model_revision'), 'kit_commit': manifest['kit_commit'],
              'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest() if protocol_path.exists() else None,
              'sample_rows_sha256': manifest['sample_rows_sha256'], 'seed': manifest['seed'],
              'requests': len(ids), 'benchmarks': len(manifest['benchmark_counts']), 'groups': manifest['group_count'],
              'all_stages_complete': all_complete, 'stages_complete': sum(r['sample_complete'] for r in rows),
              'requested_stages': requested, 'all_requested_stages_complete': requested_complete,
              'final_stage_evaluation_complete': final_complete,
              'historical_comparison_performed': len(rows) > 1 and requested_complete,
              'stages_total': len(rows), 'final_stage': 'native-bf16', 'stages': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix('.json')
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    cards = ''.join(f'<div><b>{value}</b><span>{label}</span></div>' for value, label in
                    [(len(ids), '固定请求'), (report['groups'], '完整题组'),
                     (report['benchmarks'], '公开基准'),
                     (str(report['stages_complete']) + '/' + str(len(rows)), '已测完版本')])
    table = []
    for r in rows:
        complete = 'board_method' in r
        accuracy = r.get('labelled_field_accuracy', {})
        a = f"{accuracy['accuracy']*100:.2f}%<small>{accuracy['correct']} / {accuracy['denominator']}</small>" if complete and accuracy.get('accuracy') is not None else '—'
        timing = r.get('board_method', {})
        t = f"{timing['p50_ms']:.1f} / {timing['p95_ms']:.1f}" if timing.get('count') else '—'
        state = ('已完成' if complete else '已暂停' if r['execution_status'] == 'paused_by_user'
                 else '未完成' if r['execution_status'] in ('failed', 'interrupted')
                 else '运行中' if r['execution_status'] == 'running' else '待测')
        index = f"{r['sampled_public_index']:.2f}" if r.get('sampled_public_index') is not None else '—'
        counts = r['counts']
        table.append(f'<tr><th>{html.escape(r["label"])}<small>{html.escape(r["stage"])}</small></th><td>{state}<small>{r["completed_requests"]} / {len(ids)}</small></td><td>{a}</td><td>{index}</td><td>{t}</td><td>{counts.get("error",0)} / {counts.get("unsupported",0)}</td></tr>')
    data = json.dumps(report, ensure_ascii=False).replace('<', '\\u003c')
    document = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>同一份官方样本：各阶段准确率与速度对比</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f5ef;color:#202b29;font:15px/1.6 system-ui,sans-serif}main{max-width:1180px;margin:auto;padding:52px 24px}h1{font-size:clamp(26px,4vw,44px);line-height:1.25;letter-spacing:-.04em;margin:12px 0 18px}h2{font-size:22px;margin-top:44px}p{max-width:850px;color:#5d6964}a{color:#2c7461}code{overflow-wrap:anywhere}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}.cards div,.chart{background:white;padding:20px;border-radius:12px}.cards b{display:block;font-size:32px}.cards span,small{display:block;font-size:12px;color:#6f7e76}.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}.bar{display:grid;grid-template-columns:145px 1fr 80px;gap:8px;align-items:center;margin:12px 0;font-size:11px}.track{background:#eff1ec;height:12px}.fill{display:block;height:100%;background:#267762}.bar b{text-align:right;font-size:12px}.chart h3{margin:0;font-size:17px}.chart p{font-size:12px}.scroll{overflow-x:auto;background:white;border-radius:12px}table{border-collapse:collapse;width:100%;min-width:850px;text-align:left;font-size:13px}th,td{padding:15px 13px;border-bottom:1px solid #e9ece7;vertical-align:top}thead{background:#e8ede6}tbody th{font-weight:600;width:235px}td{font-variant-numeric:tabular-nums}details{margin:22px 0;padding:18px;background:white;border-radius:12px}summary{cursor:pointer;font-weight:600}button{border:1px solid #ccd9cc;background:#fff;color:#204c40;border-radius:6px;padding:6px 12px;margin:6px;cursor:pointer}.note{background:#e3eade;border-radius:10px;padding:16px;font-size:13px}.footer{margin-top:35px;font-size:12px}@media(max-width:700px){main{padding:30px 16px}.cards{grid-template-columns:1fr 1fr}.charts{grid-template-columns:1fr}.bar{grid-template-columns:125px 1fr 70px}}
</style><main><small>DECISION INDEX 0.3 / RK3576 / 同模型与同样本</small><h1>同一份官方样本，<br>逐阶段比较准确率与速度。</h1><p>@@STATUS@@ 原始请求和选项保持不变；失败与不支持单独列出。下表不是官方完整榜单成绩。</p><div class="cards">@@CARDS@@</div><div class="charts"><section class="chart"><h3>成功请求的板端 P50</h3><p>从零开始的线性刻度，单位 ms。须与失败率一起读。</p><div id="latency-chart"></div></section><section class="chart"><h3>官方计分面板内的单标签字段正确率</h3><p>百分比刻度从 0 到 100；微平均，不能替代各任务原生指标。</p><div id="accuracy-chart"></div></section></div><h2>统一样本的结果</h2><div class="scroll"><table><thead><tr><th>版本</th><th>状态 / 请求数</th><th>单标签字段正确率</th><th>抽样综合分</th><th>P50 / P95，ms</th><th>错误 / 不支持</th></tr></thead><tbody>@@ROWS@@</tbody></table></div><details><summary>怎么读，怎么复现</summary><p>正确率分母包含有明确标签但执行失败/不支持的字段。排序、群体偏好、连续真值使用各自原生指标；SGD、RouterBench 等不计入官方当前主分的基准不混入额外字段正确率。抽样综合分按上游公式计算，但只覆盖抽中题组。</p><p>速度为模型已加载后的板端 system_one 方法墙钟。排除模型加载与初始两次统一预热，保留首次形状的准备成本；通信耗时另存。按输入 token 分组时使用 usage.input_tokens，它可能累计同一请求中多个问题的 token，不能直接当作单个序列长度。</p><p>不同版本成功集合可能不同；完整 JSON 另提供与 CPU 适配版在共同成功请求上的配对速度。未测完的版本不展示准确率结论。小样本差异不自动证明完整业务集准确率变化。</p><p>样本内容 SHA256：<code>@@HASH@@</code>；种子 20260919。<a href="@@JSON@@">查看汇总 JSON</a>。</p></details><p class="note">原始题目、标签与逐次输出留在本地评测目录中；页面只包含汇总数据。最终部署状态以实际健康核验记录为准。</p><p class="footer">固定模型 revision：20c745653400d191ecacd99c4eafc0c5900dc15e</p></main><script type="application/json" id="data">@@DATA@@</script><script>
const data=JSON.parse(document.querySelector('#data').textContent);const finished=data.stages.filter(r=>r.sample_complete&&r.board_method);function chart(id,metric,max){const root=document.getElementById(id);if(!finished.length){root.textContent='等待首个版本完成正式测试。';return}for(const row of finished){const value=metric(row);if(value===null||value===undefined)continue;const bar=document.createElement('div');bar.className='bar';const label=document.createElement('span');label.textContent=row.label;const track=document.createElement('span');track.className='track';const fill=document.createElement('i');fill.className='fill';fill.style.width=(Math.min(value/max,1)*100)+'%';track.append(fill);const number=document.createElement('b');number.textContent=value.toFixed(1);bar.append(label,track,number);root.append(bar)}}chart('latency-chart',r=>r.board_method.p50_ms,Math.max(1,...finished.map(r=>r.board_method.p50_ms)));chart('accuracy-chart',r=>r.labelled_field_accuracy.accuracy===null?null:r.labelled_field_accuracy.accuracy*100,100);
</script></html>'''
    if final_only:
        document = document.replace('同一份官方样本：各阶段准确率与速度对比', '最终优化版本：官方样本实测')
        document = document.replace('同一份官方样本，<br>逐阶段比较准确率与速度。', '最终优化版本，<br>官方样本实测。')
        document = document.replace('@@STATUS@@ 原始请求和选项保持不变；',
                                    '@@STATUS@@ 本轮只测试最终 native BF16 版本，不提供跨阶段对比结论。原始请求和选项保持不变；')
        document = document.replace('不同版本成功集合可能不同；完整 JSON 另提供与 CPU 适配版在共同成功请求上的配对速度。',
                                    '历史版本测试已按用户要求停止；之前的部分 CPU 记录独立归档，不纳入本轮。')
    paused = any(r['execution_status'] == 'paused_by_user' for r in rows)
    status_text = ('本轮指定版本已测完。' if requested_complete else '评测已暂停，先定位真实慢请求的优化点。' if paused
                   else '测试正在进行，暂不下最终结论。')
    for key, value in {'STATUS': status_text,
                       'CARDS': cards, 'ROWS': ''.join(table), 'HASH': manifest['sample_rows_sha256'],
                       'JSON': json_path.name, 'DATA': data}.items():
        document = document.replace('@@' + key + '@@', value)
    output.write_text(document)
    print(json.dumps({'complete': report['stages_complete'], 'total': len(rows), 'report': str(output)}))
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', default='output/decision-index-audit/runs')
    p.add_argument('--manifest', default='output/decision-index-audit/sample-1000.jsonl.manifest.json')
    p.add_argument('--out', default='output/decision-index-audit/comparison.html')
    a = p.parse_args()
    build(Path(a.runs), json.loads(Path(a.manifest).read_text()), Path(a.out))


if __name__ == '__main__':
    main()
