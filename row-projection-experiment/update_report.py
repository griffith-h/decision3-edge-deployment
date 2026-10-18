# -*- coding: utf-8 -*-
"""Rebuild the first-stage report, then prepend this stage without rewriting evidence."""
from pathlib import Path
import json,html,subprocess,sys,statistics
P=Path(__file__).resolve().parent;root=P.parent;old=root/'mlp-length-experiment'
load=lambda n:json.loads((P/n).read_text());esc=lambda s:html.escape(str(s));ms=lambda s:f'{1000*s:.2f}'
screen=load('logs/screen.json');f=load('logs/formal.json');d=load('logs/decomposition.json');fs=load('logs/fused-screen.json');valid=load('logs/validation.json');after=load('service-after.json');assert valid['passed'] and after['passed'] and f['passed']
fast=load('logs/fast-batch.json');assert fast['completed'];fast_records=fast['records'];fp=max(x['probability_max_delta'] for x in fast_records);fsmax=max(x['score_max_delta'] for x in fast_records);fc=max(x['confidence_max_delta'] for x in fast_records)
mode=f['candidate'];formal_rows=[];candidates=[]
for key,c in f['cases'].items():
 a,b=c['variants']['mlp_only'],c['variants'][mode];records=b['records'];candidates+=records;gain=1-b['timing']['median']/a['timing']['median'];shape=next(iter(records[0]['projections']['shapes']),next(iter(records[0]['mlp']['shapes'])))
 basepassed=all(x['passed'] for x in a['records']);baseflag='' if basepassed else '<br><b>该输入精度未通过</b>'
 p95=ms(b['timing']['p95_nearest_rank']) if len(records)>=20 else '样本少，不作尾延迟结论'
 formal_rows.append(f'<tr><td>{esc(key)}<br>usage={c["input_tokens"]}</td><td><code>{esc(shape)}</code></td><td>{ms(a["timing"]["median"])}<br>n={len(a["records"])}{baseflag}</td><td>{ms(b["timing"]["median"])}<br>n={len(records)}</td><td>{100*gain:+.2f}%</td><td>{p95}</td><td>{max(x["probability_max_delta"] for x in records):.8f}<br>{max(x["score_max_delta"] for x in records):.8f}<br>{max(x["confidence_max_delta"] for x in records):.8f}</td><td>{records[0]["projections"]["optimized_projections"]} / {records[0]["mlp_policy_fallbacks"]}</td></tr>')
candidates+=f['extra'];maxp=max(x['probability_max_delta'] for x in candidates);maxs=max(x['score_max_delta'] for x in candidates);maxn=max(x['noul_max_delta'] for x in candidates);maxc=max(x['confidence_max_delta'] for x in candidates)
initial=[]
for key,c in screen['cases'].items():
 cells=[]
 for m in ['current','mlp_only','single','chunked']:
  v=c['variants'][m];recs=v['records'];cells.append(f'<td>{ms(v["timing"]["median"])}<br>Δ预测={max(x["max_delta"] for x in recs):.8f}<br>{"通过" if v["passed"] else "未通过"}</td>')
 initial.append(f'<tr><td>{esc(key)}<br>usage={c["input_tokens"]}</td>'+''.join(cells)+'</tr>')
decomp=[]
for m,v in d['variants'].items():
 probability_delta=max(abs(value-d['reference']['answers'][q]['probabilities'][label]) for q,answer in v['output']['answers'].items() for label,value in answer.get('probabilities',{}).items())
 fields=''.join(f'<div><code>{esc(x["path"])}</code>: {x["reference"]:.8f} → {x["candidate"]:.8f}，Δ={x["delta"]:.8f}</div>' for x in v['worst_fields'][:2])
 decomp.append(f'<tr><td>{esc(m)}</td><td>{fields}</td><td>{probability_delta:.8f}</td><td>{"通过" if v["passed"] else "未通过"}</td><td>{v["mlp_policy_fallbacks"]}</td></tr>')
fusion=[]
for key,c in fs['cases'].items():
 a,b=c['variants']['chunked'],c['variants']['fused'];r=b['records'];fusion.append(f'<tr><td>{esc(key)}</td><td>{ms(a["timing"]["median"])}</td><td>{ms(b["timing"]["median"])}</td><td>{100*(1-b["timing"]["median"]/a["timing"]["median"]):+.2f}%</td><td>{r[0]["attention_fusion"]["optimized"]} / {r[0]["attention_fusion"]["fallback"]}</td><td>{"通过" if b["passed"] else "未通过"}</td></tr>')
coverage=''.join(f'<tr><td><code>{esc(x["name"])}</code></td><td>{x["k"]}</td><td>{x["n"]}</td></tr>' for x in screen['inventory']['selected'])
summary={'candidate':mode,'max_probability_delta':maxp,'max_score_delta':maxs,'max_noul_delta':maxn,'max_confidence_delta':maxc,'candidate_checked_outputs':len(candidates),'all_selected_outputs_passed':all(x['passed'] for x in candidates),'fast_batch':{'timing':fast['timing'],'max_probability_delta':fp,'max_score_delta':fsmax,'max_confidence_delta':fc,'prediction_gate_passed':fast['prediction_gate_passed'],'probability_only_gate_passed':fast['probability_only_gate_passed']},'projection_count':42,'production_unchanged':True,'cases':{k:{'mlp_only':v['variants']['mlp_only']['timing'],'candidate':v['variants'][mode]['timing'],'reduction_vs_mlp':v['reduction_vs_mlp']} for k,v in f['cases'].items()}}
(P/'logs/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
links=['screen','fused-screen','decomposition','guarded-regression','formal','fast-batch','unit','native-unit','extra-guards','artifacts','validation','summary','html-static']
linktext=' · '.join(f'<a href="../row-projection-experiment/logs/{n}.json">{n}.json</a>' for n in links)
new=f'''<section id="row-projection" class="box"><h2>第二阶段：其余大型线性投影跨512行分块</h2><p><b>42个非MLP语言投影已接入分块，原生每次仍最多512行；没有修改模型权重、非因果注意力mask、位置或扫描顺序，也没有反编译或替换线上服务。</b>先拼回完整线性输出，再执行原完整注意力/扫描。同输入组继续复用原双核并行、列打包和句柄。</p><p>并列验证快路径 <code>fused</code> 和严格score保护版 <code>{esc(mode)}</code>。下方正式对照表采用保护版：在分块之上恢复已有的注意力预处理融合；实际MLP批次B&gt;1且B×T&gt;512时，保留原MLP，以处理本轮发现的score偏差。115及512以内的投影仍走原路径。适配保护范围为512&lt;B×T≤2048，更长或未知签名继续原路径，不截断。</p>
<h3>正式计时与精度</h3><p>单位ms；准确性都对照同进程重新计算的current（线上fixed115等价路径），速度对照上一阶段mlp_only。115-token为20组交替配对；513/615行分别5次MLP-only控制与20次候选，分5段交错；多问题保护路径各5次配对。首次准备/暖机不计入分布，常驻矩阵句柄数在计时期间不增加。不是HTTP/排队或并发SLA。</p><div class="scroll"><table><thead><tr><th>输入</th><th>实际张量示例</th><th>MLP-only P50</th><th>本轮候选P50</th><th>相对MLP-only延迟下降</th><th>候选P95</th><th>Δ概率<br>Δscore<br>Δconfidence</th><th>非MLP加速数 / 原MLP层数</th></tr></thead><tbody>{''.join(formal_rows)}</tbody></table></div>
<p><b>受检候选与24次额外短请求回归：</b>共{len(candidates)}次；概率最大绝对差{maxp:.8f}，score最大差{maxs:.8f}，noul最大差{maxn:.8f}，confidence最大差{maxc:.8f}。choice均一致，原有概率/score/noul≤0.01门槛通过；confidence单独列出，没有将其计入0.01通过结论。没有标注业务数据，不能宣称业务准确率无损，也不能保证每个新输入都≤0.01。</p>
<h3>多问题快路径：20次单独复核</h3><p>同一跨512多问题输入，快路径中位 <b>{ms(fast["timing"]["median"])} ms</b>、P95 {ms(fast["timing"]["p95_nearest_rank"])} ms，n=20；概率最大差 <b>{fp:.8f}</b>，score最大差 <b>{fsmax:.8f}</b>，confidence最大差 <b>{fc:.8f}</b>，choice全部一致。它满足仅看概率≤0.01的条件，仍未满足原来的score绝对差≤0.01条件。不能把这两个判断混成“精度全部通过”，也不能把score的0.01745当作概率百分点。</p><p>这组复核使用正式对照中同一冻结模型current输出作为评估参照；候选仍实跑20次，无输入/答案缓存。保守版沿用原score门槛，在这个样本上可能慢于激进MLP-only和快路径；两种结果都保留供业务阈值取舍，本轮未替换服务。</p><h3>为什么保留多问题MLP保护</h3><p>新增跨512多问题样本的实际MLP形状为[3,239,1024]、717行，usage为680。原始MLP-only已出现complexity.score差0.01312654；全量组合差0.01745270，均超过沿用的绝对0.01门槛。score原量程为0–2，不能将其直接当作分类概率百分点。以下保留失败路线，不能按速度将它们当作通过。</p><div class="scroll"><table><thead><tr><th>模式</th><th>误差最大字段（含confidence）</th><th>仅分类概率最大Δ</th><th>原完整精度门槛</th><th>保留原MLP层数</th></tr></thead><tbody>{''.join(decomp)}</tbody></table></div><p>该样本保护后score差约0.00461024，confidence差约0.00617638；42个非MLP投影仍全部按行分块，保留的CPU调用仅来自原MLP的36个线性投影。另加自然语言B=2跨512输入，两次保护版回归也通过，见guarded-regression.json。保护版可能慢于未通过门槛的激进MLP-only/全量图路线；速度、精度两种结果都保留。</p>
<h3>分步筛选：每模式两次，含首次路径准备</h3><p>current是线上等价路径；mlp_only是上轮方案；single逐投影拆块；chunked保留同层投影成组并行。此筛选各模式两次，含可能首次句柄准备，不能当作正式P95，也不与后面正式分布混用。Δ预测包含概率/score/noul，confidence另见原始JSON。</p><div class="scroll"><table><thead><tr><th>输入</th><th>current ms</th><th>mlp_only ms</th><th>single ms</th><th>chunked ms</th></tr></thead><tbody>{''.join(initial)}</tbody></table></div>
<h3>恢复已有融合的单独对照</h3><p>512判断还关闭了已有CPU注意力预处理融合。本轮只给这一入口提供可分块的group门面，原engine.eligible和raw MLP pipeline上限仍512；B=1等已有保护保持。每模式4次、提前暖机。本表的fused尚不带多问题MLP保护，故失败多问题样本仍标为未通过。</p><div class="scroll"><table><thead><tr><th>输入</th><th>chunked ms</th><th>fused ms</th><th>下降</th><th>融合/回退层数</th><th>精度</th></tr></thead><tbody>{''.join(fusion)}</tbody></table></div>
<h3>覆盖、回退与服务保护</h3><p>所有正常跨512候选的42个目标投影均有覆盖，每次原生rows≤512；不存在因512尺寸而回退的非MLP目标投影。小型门控、视觉投影、分类头以及MLP不在这42个集合里，分别沿既有路径和本阶段MLP策略运行。范围失败丢弃全部局部输出并整组CPU重放；运行失败先等待原组工作线程完成，再整组重放并禁用故障成员。保护检查覆盖权重/偏置版本、钩子、训练/梯度、dtype、非连续和跨批次行序。</p><details><summary>42个目标投影与矩阵维度</summary><div class="scroll"><table><thead><tr><th>完整名称</th><th>输入K</th><th>输出N</th></tr></thead><tbody>{coverage}</tbody></table></div></details><p>聚焦单测、真实NPU列顺序/偏置/跨批次测试、记录完整性与句柄复用校验通过。线上容器仍为<code>{esc(after['id'][:12])}</code>，启动时间、配置与{len(after['files'])}个发布文件哈希不变；五次HTTP完整输出与实验前相等，健康且测试容器退出。少量首尾HTTP样例不用于宣称前后性能等价。</p>
<h3>复现与原始证据</h3><p>详见<a href="../row-projection-experiment/REPRODUCTION.md">本阶段复现步骤</a>、<a href="../row-projection-experiment/DESIGN.txt">设计验收</a>、<a href="../row-projection-experiment/VERIFY.txt">验证命令与结果</a>。新目录独立无端口容器，发布和上轮MLP目录只读；没有构建镜像、重新转换权重或推送部署。</p><pre>bash run.sh test_guards.py
bash run.sh test_native.py
bash run.sh screen.py
bash run.sh decompose.py
bash run.sh screen_fused.py
bash run.sh formal.py --candidate fused_guarded
bash run.sh formal_fast_batch.py
# 板端测完退出后，本机顺序执行，哈希不与计时重叠：
python3 row-projection-experiment/preserve_service.py after
python3 row-projection-experiment/capture_artifacts.py
python3 row-projection-experiment/validate_records.py
python3 row-projection-experiment/update_report.py</pre><p>{linktext}</p><p><a href="../row-projection-experiment/service-before.json">本阶段service-before</a> · <a href="../row-projection-experiment/service-after.json">本阶段service-after</a></p><p><b>本轮仍没有通用200ms结果。</b>新增跨512输入的projection回退已在上述范围解决；短请求原路径保持，多问题保护的剩余CPU MLP仍有代价。另有两个独立资格限制：CPU完整注意力和GatedDeltaNet块扫描仅在序列T≤512启用优化，T>512仍回到已有Torch/FP32备用路径；这与本次已处理的投影B×T上限不同。正式记录cpu_kernel_delta保存其优化/回退次数。本轮没有放宽这两种内核的范围或改动扫描状态传递。未测试视频、并发吞吐、其他芯片或更换模型。</p></section><div class="box note"><b>以下是第一阶段MLP图实验原始记录。</b>其中“本轮”指第一阶段。新增投影分块及精度边界请以上方第二阶段为准。</div>'''
subprocess.run([sys.executable,str(old/'build_report.py')],stdout=subprocess.DEVNULL,check=True)
path=old/'report.html';s=path.read_text();s=s.replace('d3-edge 多长度 MLP 图实验','d3-edge 多长度 MLP 与投影分块实验',1).replace('让不同长度的请求复用 MLP 图','多长度 MLP 与线性投影加速记录',1).replace('<nav>','<nav><a href="#row-projection">本轮投影分块</a>',1).replace('</nav>','</nav>'+new,1)
s=s.replace('下一步可单独验证其他线性投影按行分块，或调整公开矩阵接口的适配范围，尝试消除超过512行的回退；本轮未实现或测量这一步，不能承诺其耗时或数值一致性。','第一阶段提出的投影按行分块已在上方第二阶段验证；其数据、范围和精度保护以上方记录为准。')
s=s.replace('报告由build_report.py从原始JSON生成。','第一阶段由build_report.py生成；第二阶段由row-projection-experiment/update_report.py从原始JSON整合。')
path.write_text(s);print('UPDATED',path)
