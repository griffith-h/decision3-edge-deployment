"""Audit the fixed-request diagnostic and publish only aggregate timing data."""
import argparse,collections,hashlib,html,json,statistics
from pathlib import Path
from request_cost_ledger import partition,union
from summarize import compare

LABELS={
 'sdk_runtime_sync':'SDK同步调用覆盖的调用/等待区间','cpu_scan_matmul':'CPU扫描：矩阵乘法',
 'cpu_scan_triangular_solve':'CPU扫描：三角求解','cpu_scan_copy_convert':'CPU扫描：显式复制/转换',
 'cpu_scan':'CPU扫描：其余运算/循环','cpu_attention':'CPU完整注意力',
 'repair_dots':'精度修复：选点重算','repair_norms':'精度修复：范数准备','repair_mask':'精度修复：筛选',
 'repair_other':'精度修复：其余舍入/索引/检查','fp32_read':'FP32结果读取/布局转换','fp32_read_other':'读取外围分配/检查',
 'projection_native_call':'投影桥接：SDK之外转换/准备','mlp_native_call':'MLP桥接：SDK之外转换/准备',
 'mlp_product_prepare':'MLP激活/乘积/原生输入准备','projection_other':'投影调度/拼接/其余处理',
 'mlp_other':'MLP分块/分配/其余处理','worker_join':'等待worker：非SDK部分（含结果转换）',
 'cpu_convolution':'CPU卷积','normalization':'归一化','scan_other':'扫描层外围运算',
 'attention_other':'注意力层外围运算','decoder_other':'decoder其余运算',
 'model_other':'模型其余处理','request_prepare':'请求准备/分词','request_respond':'响应计算/组装','request_other':'请求其余处理'}

def load(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def audit(folder):
    base=folder.parent.parent;data=load(folder/'request-cost.json');execution=load(folder/'execution.json')
    assert execution['exit_code']==0 and execution['request_completed'] and execution['retained_healthy'] and not execution['cleanup_errors']
    assert not execution.get('budget_exceeded') and not execution['worker_final_state']['OOMKilled']
    assert data['completed'] and len(data['records'])==6
    request=next(r['payload'] for r in load(base/'r018/inputs.json') if r['id']=='diagnostic-home-pair')
    assert hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==data['payload_sha256']
    assert digest(Path(__file__).with_name('check_regression.py'))=='841c6b517241d3d9af051bcd18a2f51a83c2d30a4aef4b05eaec1e85f1195683'
    assert load(folder/'lab-sources.json')==load(folder.parent/'candidate-sources.json')['sources']
    original_sources=load(base/'r080/20261020T023644-context-regression34-full-batch-sparse-request-scope-plain/lab-sources.json')
    differences=[name for name,value in original_sources.items() if name in load(folder/'lab-sources.json') and load(folder/'lab-sources.json')[name]!=value]
    assert set(differences)<=set(('probe.py','host_run.py')),(differences,'Unexpected qualified-source change')
    backend=load(folder/'runtime-backend-before-cleanup.json')
    assert backend['mkldnn'] and backend['threads']==3 and backend['float32_precision']=='highest' and not backend['autocast_cpu']
    assert all(v==0 for v in load(folder/'lifecycle.json').values())
    tests=load(folder/'isolated-boundary-checks.json');assert tests['passed']
    cpu=load(base/'r061/20261019T170841-diagnostic-home-pair-cpu-full-plain/result.json')['response']
    qualified=load(base/'r072/20261020T004105-diagnostic-home-pair-full-batch-sparse-band25-plain/result.json')['response']
    comparisons=[];traces=[]
    for r in [data['warmup']]+data['records']:
        c=compare(r['response'],cpu,False)
        assert c['max_delta']<=.01 and all(c[k] for k in ('choices_unchanged','noul_predictions_unchanged','usage_equal','model_equal'))
        assert r['response']==qualified
        assert r['counter_delta']==data['records'][0]['counter_delta']
        comparisons.append(c)
        if r['mode']=='ledger':
            trace=load(folder/r['trace_file']);actual=partition(trace['events'],trace['root'],trace['sdk'])
            assert actual==trace['ledger'] and actual['closure_error_ns']==0 and actual['exclusive_seconds']==r['exclusive_seconds']
            assert len(trace['sdk'])==r['sdk_calls']>0 and all(e['status']==0 for e in trace['sdk'])
            assert all(not e.get('error') for e in trace['events'])
            calls=collections.Counter(e['category'] for e in trace['events'])
            assert calls['cpu_scan']==calls['cpu_attention']==6 and calls['cpu_scan_matmul']==234 and calls['cpu_scan_triangular_solve']==12
            traces.append(trace)
    plain=[r for r in data['records'] if r['mode']=='plain'];profile=[r for r in data['records'] if r['mode']=='ledger']
    mean_plain=statistics.mean(r['seconds'] for r in plain);mean_profile=statistics.mean(r['seconds'] for r in profile)
    categories=sorted({k for r in profile for k in r['exclusive_seconds']})
    rows=[{'category':k,'label':LABELS[k],'measured_profile_mean_seconds':statistics.mean(r['exclusive_seconds'].get(k,0) for r in profile)} for k in categories]
    rows.sort(key=lambda r:-r['measured_profile_mean_seconds'])
    assert abs(sum(r['measured_profile_mean_seconds'] for r in rows)-mean_profile)<1e-9
    factor=mean_plain/mean_profile
    for r in rows:
        r['share']=r['measured_profile_mean_seconds']/mean_profile
        r['normalized_plain_estimate_seconds']=r['measured_profile_mean_seconds']*factor
    values={r['category']:r['normalized_plain_estimate_seconds'] for r in rows}
    directions=[]
    groups=[('精度修复及FP32读取',('repair_dots','repair_norms','repair_mask','repair_other','fp32_read','fp32_read_other')),
            ('CPU扫描整体',('cpu_scan','cpu_scan_matmul','cpu_scan_triangular_solve','cpu_scan_copy_convert')),
            ('其中：扫描矩阵乘法',('cpu_scan_matmul',)),
            ('其中：修复选点重算',('repair_dots',)),
            ('CPU完整注意力',('cpu_attention',)),
            ('SDK覆盖的调用/等待区间',('sdk_runtime_sync',)),
            ('原生转换/准备',('projection_native_call','mlp_native_call','mlp_product_prepare'))]
    for label,keys in groups:
        cost=sum(values.get(k,0) for k in keys)
        directions.append({'direction':label,'estimated_current_seconds':cost,'saving_if_25pct_less':cost*.25,
                           'saving_if_50pct_less':cost*.5,'elimination_ceiling_seconds':cost})
    sdk_sums=[sum(e['end_ns']-e['start_ns'] for e in t['sdk'])/1e9 for t in traces]
    sdk_unions=[sum(b-a for a,b in union((e['start_ns'],e['end_ns']) for e in t['sdk']))/1e9 for t in traces]
    result={'passed':True,'request_payload_sha256':data['payload_sha256'],'source_differences_from_r080':differences,
            'cases':'B2/T1193,2 questions with lengths1193/1102;2295 effective and2386 padded tokens; unchanged padding/batch/model',
            'mean_plain_seconds':mean_plain,'median_plain_seconds':statistics.median(r['seconds'] for r in plain),
            'plain_seconds':[r['seconds'] for r in plain],'profile_seconds':[r['seconds'] for r in profile],
            'mean_profile_seconds':mean_profile,'instrumentation_ratios':data['paired_instrumentation_ratios'],
            'preserved_warmup_seconds':data['warmup']['seconds'],
            'median_instrumentation_ratio':data['median_instrumentation_ratio'],
            'mean_profile_process_cpu_seconds':statistics.mean(r['process_cpu_seconds'] for r in profile),
            'mean_sdk_thread_sum_seconds':statistics.mean(sdk_sums),'mean_sdk_wall_union_seconds':statistics.mean(sdk_unions),
            'sdk_calls_per_profile':[len(t['sdk']) for t in traces],
            'sdk_mean_host_thread_cpu_seconds':statistics.mean(sum(e['cpu_end_ns']-e['cpu_start_ns'] for e in t['sdk'])/1e9 for t in traces),
            'max_cpu_nonconfidence_delta':max(c['max_delta'] for c in comparisons),'all_response_dictionaries_equal_qualified':True,
            'arm_checks':tests['tests_run'],'rows':rows,'directions':directions,'worker_spans_are_separate_overlapping_view':True,
            'wall_closure_error_ns':0,'estimation_boundary':'Normalized plain estimates assume profile proportions transfer.25/50pct and elimination are scenarios, not measured attainable gains. Overlapping directions must not be added.'}
    (folder/'request-cost-review.json').write_text(json.dumps(result,indent=2)+'\n');return result

def render(result,target):
    r=result;table=['| 环节 | 插桩实测均值 | 占比 | 无插桩等比例估算 |','| --- | ---: | ---: | ---: |']
    cost={d['direction']:d['estimated_current_seconds'] for d in r['directions']}
    major=cost['精度修复及FP32读取']+cost['CPU扫描整体']+cost['CPU完整注意力']
    remaining=r['mean_plain_seconds']-major-cost['SDK覆盖的调用/等待区间']
    groups=[('精度修复及FP32读取',cost['精度修复及FP32读取']),('CPU扫描',cost['CPU扫描整体']),
            ('CPU完整注意力',cost['CPU完整注意力']),('SDK覆盖区间',cost['SDK覆盖的调用/等待区间']),('其他处理及worker剩余等待',remaining)]
    for row in r['rows']:table.append(f"| {row['label']} | {row['measured_profile_mean_seconds']:.4f}秒 | {row['share']:.2%} | {row['normalized_plain_estimate_seconds']:.4f}秒 |")
    scenarios=['| 方向（存在包含关系，不可相加） | 当前估算 | 减少25%可省 | 减少50%可省 | 完全消除的上限 |','| --- | ---: | ---: | ---: | ---: |']
    for d in r['directions']:scenarios.append('| '+d['direction']+' | '+' | '.join(f"{d[k]:.3f}秒" for k in ('estimated_current_seconds','saving_if_25pct_less','saving_if_50pct_less','elimination_ceiling_seconds'))+' |')
    md=f'''# 约31秒请求的完整耗时账本

2026-10-20，RK3576，r087诊断；模型与r080数值合格计算路径不变。本报告不改计算策略，不代表HTTP、并发或完整官方样本成绩。

## 结论与优化优先级

**主要成本是CPU端的数值修复、扫描和完整注意力，合计约{major/r['mean_plain_seconds']:.2%}。** 固定中长度路径为满足原数值门槛保留原CPU扫描/注意力，并对NPU投影/MLP结果做CPU临界选点修复。这里定位的是当前软件路径的成本，不是硬件能力上限。

优先研究精度修复中的选点重算：当前归一化估算{cost['其中：修复选点重算']:.3f}秒。保持同一选点集合、原2.5e-7保护带和每个输出的K归约顺序，先检查独立输出间是否能复用加载/转换及改善指令流水；不减少修复点、不跳算、不降低门槛。若该部分减少25%，整请求条件估算可省{cost['其中：修复选点重算']*.25:.3f}秒；能否做到尚未验证。读取与筛选融合是同组的另一候选，须先测数据流，不能把两者上限相加当已获收益。

第二优先是CPU完整注意力（约{cost['CPU完整注意力']:.3f}秒）与扫描矩阵乘法（约{cost['其中：扫描矩阵乘法']:.3f}秒），先看保持原数值语义的数据布局/复用方案。已验证变慢的整段扫描单线程不再重复；原生扫描更改答案的方案继续拒绝。

SDK覆盖区间仅约{cost['SDK覆盖的调用/等待区间']:.3f}秒，即便理想消除也不足以将31秒降到15秒。三大CPU成本即便都减少一半，条件估算整请求仍约{r['mean_plain_seconds']-major*.5:.2f}秒。下一轮先证明约1秒级整请求潜力再进入整模型配对；新候选仍须完整34例、保护1e-6及完整18题原门槛，不能用账本代替验收。

## 请求与验证

- 同一请求：2个问题，长度1193/1102，B2/T1193，共2295有效token、补齐后2386行，原批次/padding保持；请求载荷SHA256：`{r['request_payload_sha256']}`。
- 形状预热{r['preserved_warmup_seconds']:.4f}秒完整保留，不计为配对结果；再做3组无插桩/插桩AB/BA。无插桩耗时：{', '.join(f'{x:.4f}' for x in r['plain_seconds'])}秒；中位{r['median_plain_seconds']:.4f}秒、均值{r['mean_plain_seconds']:.4f}秒。
- 插桩耗时：{', '.join(f'{x:.4f}' for x in r['profile_seconds'])}秒；均值{r['mean_profile_seconds']:.4f}秒。配对中位耗时比{r['median_instrumentation_ratio']:.6f}。只有3组，不能据此证明微小差异或统计SLA。
- 全部响应字典与合格r072相同；对独立原CPU最大非confidence差{r['max_cpu_nonconfidence_delta']:.10f}，原0.01及choice/noul/usage/model门槛通过，confidence仍单列。所有调用/选点计数一致；后端True/3/highest，{r['arm_checks']}项ARM检查通过，关闭0、OOMFalse、原服务健康。

## 完整主线程账本

每次请求的主线程叶区间完整覆盖开始到返回，闭合误差0ns。下面使用3次均值，因此各行可相加得到插桩总时长。worker耗时不再额外相加。最后一列仅按无插桩/插桩均值比例归一化，用于收益估算，不是直接测出的无插桩组件耗时。

{chr(10).join(table)}

合计：插桩{r['mean_profile_seconds']:.4f}秒；归一化无插桩估算{r['mean_plain_seconds']:.4f}秒。

## 并行和SDK视图

- 每次SDK调用数：{r['sdk_calls_per_profile']}；状态均0。跨线程同步调用时长之和均值{r['mean_sdk_thread_sum_seconds']:.4f}秒，墙钟区间并集{r['mean_sdk_wall_union_seconds']:.4f}秒，两者不能相加。
- SDK计时是公开rknn_matmul_run入口到返回，包含提交、驱动与等待，**不是NPU纯设备计算时间**。主账本只计它与原生调用/worker等待叶区间的交集；并行worker的CPU工作可能同时发生，这不是逐任务关键路径归因，也不能保证消除SDK耗时就等量节省。
- 插桩请求进程CPU时间均值{r['mean_profile_process_cpu_seconds']:.4f} CPU秒；SDK调用线程CPU时间之和均值{r['sdk_mean_host_thread_cpu_seconds']:.4f} CPU秒。CPU秒跨线程累计，不能与墙钟秒相加，也不能直接当作可省时间。
- CPU扫描覆盖全部6次实际调用，矩阵乘法、三角求解和显式复制/转换另列；不把首次扫描结果外推全模型。频率/温度在请求前后记录，没有持续硬件性能计数，不能据此断言内存带宽已满或到达硬件极限。

## 各方向最多能省多少

以下25%/50%是条件假设，完全消除是乐观上限，均不是已实现或保证可实现的收益；包含关系与共享资源影响使各方向不能直接相加。

{chr(10).join(scenarios)}

## 复现与原始证据

命令：`caffeinate -i output/decision-index-audit/venv/bin/python official-long-input-experiment/execute.py --iteration r087 --case request-cost-ledger --variant full-batch-sparse-request-scope --budget 120`。

板端独立目录先按`build_request_sdk_trace.sh`构建诊断库；只有本诊断入口启用预加载，普通回归/服务不加载。模型/SDK/历史依赖不修改。完整载荷、输出、线程事件、SDK事件、源清单和审核仅在忽略目录`output/decision-index-audit/iterations/r087/`。账本代码见`request_cost_ledger.py`、`request_cost_probe.py`，独立审核见`build_request_cost_report.py`。
'''
    outcome_md='';outcome_html=''
    outcome_file=Path(__file__).parent/'rejected-selected-dot-ilp/summary.json'
    if outcome_file.exists():
        outcome=load(outcome_file)
        ratios='、'.join(f"{v['width']}路组件加权慢约{(v['weighted_component_ratio']-1)*100:.2f}%" for v in outcome['widths'])
        text=f"后续r088已完成首项验证：{ratios}。真实{outcome['actual_dot_calls']}次、{outcome['actual_selected']:,}选点数值检查及{outcome['arm_checks']}项ARM检查通过，但没有省1秒证据，已在组件阶段停止；未跑候选整请求、34/18题或部署。"
        outcome_md='## 账本之后的首项实验结果\n\n'+text+'\n\n详见[归档](rejected-selected-dot-ilp/README.md)。\n\n'
        outcome_html='<section><h2>首项验证结果：无收益，已停止</h2><p>'+html.escape(text)+'</p><a href="rejected-selected-dot-ilp/README.md">负面结果与复现</a></section>'
    if (Path(__file__).parent/'SELECTED_DOT_DIAGNOSIS.html').exists():
        outcome_md+='## 后续诊断：为何8/16路变慢\n\nr089/r090固定二进制的1040个PMU窗口及2×2对照表明，权重行步幅会放大L1缓存回填；16路另有尾部重复计算，K热循环没有落栈。未采用新布局，未证明整请求省1秒。见[诊断图表](SELECTED_DOT_DIAGNOSIS.html)与[数值、方法边界](SELECTED_DOT_DIAGNOSIS.md)。\n\n'
        outcome_html+='<section><h2>为什么越并行反而越慢？</h2><p>已补齐1040个PMU窗口和行步幅×对齐的2×2对照：主要证据指向L1回填放大，16路另有尾部重复计算；K热循环无落栈。仅诊断，未采用新布局。</p><a href="SELECTED_DOT_DIAGNOSIS.html">查看原因、计数器证据与下一步</a></section>'
    md=md.replace('## 复现与原始证据',outcome_md+'## 复现与原始证据')
    target.write_text(md)
    colors=['#2563eb','#16a34a','#ea580c','#9333ea','#0891b2','#db2777']
    blocks=[];legend=[];group_legend=[]
    for i,(label,sec) in enumerate(groups):
        color=colors[i%len(colors)];share=sec/r['mean_plain_seconds']
        blocks.append(f'<div style="width:{share*100:.6f}%;background:{color}" title="{html.escape(label)}: 归一化估算{sec:.3f}秒"></div>')
        group_legend.append(f'<tr><td><span style="color:{color}">●</span> {html.escape(label)}</td><td>{sec:.3f}s</td><td>{share:.2%}</td></tr>')
    for i,row in enumerate(r['rows']):
        color=colors[i%len(colors)];label=html.escape(row['label']);sec=row['measured_profile_mean_seconds']
        legend.append(f'<tr><td><span style="color:{color}">●</span> {label}</td><td>{sec:.4f}s</td><td>{row["share"]:.2%}</td></tr>')
    scenario_rows=''.join('<tr><td>'+html.escape(d['direction'])+'</td>'+''.join(f'<td>{d[k]:.3f}s</td>' for k in ('estimated_current_seconds','saving_if_25pct_less','saving_if_50pct_less','elimination_ceiling_seconds'))+'</tr>' for d in r['directions'])
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>31秒请求耗时账本</title><style>body{{font:16px/1.7 system-ui,sans-serif;color:#172033;background:#f5f7fb;margin:0}}main{{max-width:1060px;margin:auto;padding:28px}}h1{{font-size:30px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card,section{{background:white;border-radius:12px;padding:20px;margin:16px 0}}.card{{flex:1;min-width:180px}}strong{{font-size:26px}}.bar{{display:flex;height:48px;border-radius:8px;overflow:hidden;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:7px 10px;border-bottom:1px solid #e5e7eb;text-align:right}}th:first-child,td:first-child{{text-align:left}}.note{{color:#556174;font-size:14px}}a{{color:#2563eb}}.scroll{{overflow-x:auto}}</style><main><h1>31秒花在哪里？</h1><p>固定2问题 · 长度1193/1102 · 原padding保持 · RK3576 · 2026-10-20 · 诊断，不改模型与数值门槛</p><div class="cards"><div class="card">无插桩中位耗时<br><strong>{r['median_plain_seconds']:.2f}s</strong></div><div class="card">账本闭合误差<br><strong>0ns</strong></div><div class="card">输出与合格版<br><strong>完全一致</strong></div></div><section><h2>CPU修复、扫描、注意力占{major/r['mean_plain_seconds']:.1%}</h2><p class="note">下图按无插桩均值{r['mean_plain_seconds']:.4f}秒归一化估算。SDK/worker重叠时间不重复相加。</p><div class="bar">{''.join(blocks)}</div><table><tr><th>环节</th><th>归一化估算</th><th>占比</th></tr>{''.join(group_legend)}</table><details><summary>展开全部27项实测账本</summary><p class="note">3次插桩均值，共{r['mean_profile_seconds']:.4f}秒；以下原始实测可相加闭合总时长。</p><table><tr><th>环节</th><th>实测均值</th><th>占比</th></tr>{''.join(legend)}</table></details></section><section><h2>优化收益的条件估算</h2><p class="note">按无插桩均值归一化；25%/50%是情景假设，最右是完全消除的乐观上限。方向有包含关系，不能相加，不是已取得收益。</p><div class="scroll"><table><tr><th>方向</th><th>当前估算</th><th>减少25%可省</th><th>减少50%可省</th><th>完全消除上限</th></tr>{scenario_rows}</table></div></section><section><h2>下一步：先攻选点重算</h2><p>约{cost['其中：修复选点重算']:.2f}秒。保持同一选点与K归约顺序，先验证复用加载/转换和指令流水；若该部分减少25%，整请求条件估算可省{cost['其中：修复选点重算']*.25:.2f}秒。</p><p class="note">尚未实施，不承诺收益。不重复已变慢的扫描单线程；新候选仍须完整34例/保护1e-6和完整18题原门槛。</p></section><section><h2>证据边界</h2><p>配对中位插桩/无插桩耗时比{r['median_instrumentation_ratio']:.6f}。全部响应字典相同，原CPU非confidence差{r['max_cpu_nonconfidence_delta']:.8f} ≤ 0.01；{r['arm_checks']}项ARM检查通过。</p><p>SDK测量包含提交、驱动和等待，不是NPU纯设备时间。3组诊断不代表完整官方样本、HTTP或并发SLA；尚未证明内存带宽或硬件极限。</p><a href="{html.escape(target.name)}">完整Markdown、原始对照及复现说明</a></section></main></html>'''
    page=page.replace('下一步：先攻选点重算','账本指向：选点重算')
    if outcome_html:
        page=page.replace('尚未实施，不承诺收益。','这是账本的条件估算；首项实验结果见下方。')
        page=page.replace('<section><h2>证据边界</h2>',outcome_html+'<section><h2>证据边界</h2>')
    target.with_suffix('.html').write_text(page)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).with_name('REQUEST_COST.md'))
    a=p.parse_args();r=audit(a.folder);render(r,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('rows','directions')},indent=2))
