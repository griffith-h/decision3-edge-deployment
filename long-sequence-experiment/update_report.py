# -*- coding: utf-8 -*-
from pathlib import Path
import json,html,subprocess,sys,statistics
P=Path(__file__).resolve().parent;root=P.parent
load=lambda name:json.loads((P/name).read_text());esc=lambda v:html.escape(str(v));ms=lambda s:f'{s*1000:.2f}'
f=load('logs/formal.json');s=load('logs/screen.json');v=load('logs/validation.json');a=load('service-after.json');u=load('logs/unit.json');assert f['completed'] and v['passed'] and a['passed']
rows=[];checked=[];summary={}
for key,c in f['cases'].items():
    base,new=c['variants']['baseline'],c['variants']['both'];recs=new['records'];checked+=recs
    shape=next(iter(recs[0]['mlp']['shapes']),'未记录');longcounts=recs[0]['long_kernels'];projection=recs[0]['projections']['optimized_projections']
    bm=max(r['vs_baseline']['max_delta'] for r in recs);cm=max(r['vs_current']['max_delta'] for r in recs);confidence=max(r['vs_current']['confidence_max_delta'] for r in recs);baseconfidence=max(r['vs_baseline']['confidence_max_delta'] for r in recs)
    p95=ms(new['timing']['p95_nearest_rank']) if len(recs)>=20 else '样本较少，不作尾延迟结论'
    rows.append(f'<tr><td>{esc(key)}<br>usage={c["input_tokens"]}<br><code>{esc(shape)}</code></td><td>{ms(base["timing"]["median"])}<br>n={len(base["records"])}</td><td>{ms(new["timing"]["median"])}<br>n={len(recs)}</td><td>{c["latency_reduction"]*100:+.2f}%</td><td>{p95}</td><td>{bm:.8f}<br>{"通过" if new["baseline_gate_passed"] else "未通过"}<br>Δconfidence={baseconfidence:.8f}</td><td>{cm:.8f}<br>{"通过" if new["current_gate_passed"] else "未通过"}<br>Δconfidence={confidence:.8f}</td><td>{longcounts["attention"]["optimized"]} / {longcounts["scan"]["optimized"]}<br>投影={projection}</td></tr>')
    summary[key]={'input_tokens':c['input_tokens'],'baseline':base['timing'],'both':new['timing'],'latency_reduction':c['latency_reduction'],'baseline_gate_passed':new['baseline_gate_passed'],'current_gate_passed':new['current_gate_passed'],'max_incremental_prediction_delta':bm,'max_current_prediction_delta':cm}
current_checked=list(checked)
checked+=f['extra']
typed=[]
for key,c in f['cases'].items():
    r=c['variants']['both']['records']
    vals=[max(x['vs_current'][name] for x in r) for name in ('probability_max_delta','score_max_delta','noul_max_delta','confidence_max_delta')]
    present=[any(field in answer for answer in c['current_reference']['answers'].values()) for field in ('probabilities','score','noul','confidence')]
    has_choice=any('choice' in answer for answer in c['current_reference']['answers'].values())
    choice_text=('一致' if all(x['vs_current']['choices_unchanged'] for x in r) else '不一致') if has_choice else '—（无此字段）'
    typed.append(f'<tr><td>{esc(key)}</td>'+''.join(f'<td>{x:.8f}</td>' if exists else '<td>—（无此字段）</td>' for x,exists in zip(vals,present))+f'<td>{choice_text}</td></tr>')
screenrows=[]
for key,c in s['cases'].items():
    cells=[]
    for mode,vv in c['variants'].items():
        rr=vv['records'];delta=max(x['vs_baseline']['max_delta'] for x in rr)
        cells.append(f'<td>{ms(vv["timing"]["median"])}<br>Δ本轮={delta:.8f}</td>')
    screenrows.append(f'<tr><td>{esc(key)}<br>usage={c["input_tokens"]}</td>'+''.join(cells)+'</tr>')
profile=[]
for mode in ('baseline','both'):
    recs=[r for r in f['cases']['words-512']['profile'] if r['mode']==mode]
    total=statistics.median(r['seconds'] for r in recs);att=statistics.median(r['long_kernels']['attention']['seconds'] for r in recs);scan=statistics.median(r['long_kernels']['scan']['seconds'] for r in recs)
    other=statistics.median(r['seconds']-sum(x['seconds'] for x in r['long_kernels'].values()) for r in recs)
    profile.append(f'<tr><td>{mode}</td><td>{ms(total)}</td><td>{ms(att)}</td><td>{ms(scan)}</td><td>{ms(other)}</td></tr>')
allcurrent=all(x['vs_current']['passed'] for x in current_checked);allincremental=all(x['vs_baseline']['passed'] for x in checked)
incfailed=', '.join(v['incremental_prediction_gate_failed_cases']) or '无';curfailed=', '.join(v['current_prediction_gate_failed_cases']) or '无'
details={'cases':summary,'candidate_outputs':len(checked),'independent_current_comparisons':len(current_checked),'incremental_gate_all_passed':allincremental,'current_gate_all_passed':allcurrent,'max_probability_delta_current':max(r['vs_current']['probability_max_delta'] for r in current_checked),'max_score_delta_current':max(r['vs_current']['score_max_delta'] for r in current_checked),'max_confidence_delta_current':max(r['vs_current']['confidence_max_delta'] for r in current_checked),'production_unchanged':True}
(P/'logs/summary.json').write_text(json.dumps(details,ensure_ascii=False,indent=2)+'\n')
links=' · '.join(f'<a href="../long-sequence-experiment/logs/{name}.json">{name}.json</a>' for name in ('unit','screen','formal','artifacts','validation','summary'))
section=f'''<section id="long-sequence" class="box"><h2>第三阶段：长序列完整注意力与GatedDeltaNet扫描</h2>
<p><b>已将我们自有CPU内核的长序列路径扩到512&lt;T≤2048；模型级测试中，6层完整注意力和6层扫描均命中扩展路径，没有因512长度限制而回退。</b>未修改模型权重、层数、mask、位置或厂商库，线上服务保留。新增加速发生在CPU；NPU投影与MLP仍使用前两阶段方案，不能称为RKLLM原生支持。</p>
<p>完整注意力一次处理全部K/V，保留非因果访问及布尔padding mask；GatedDeltaNet仍使用原32行块/64列tile，状态在一次完整序列调用中跨块传递。没有把序列切成多个相互独立的512行请求。T≤512仍委托原路径，T&gt;2048及未知签名保留原回退；2048是本阶段保护范围，不是硬件或模型上限。完整模型实测范围以下表为准，不能将内核2048边界单测说成完整模型已验证到2048。</p>
<h3>正式计时：对照上一阶段投影分块版</h3><p>单位ms，均为暖机后的模型方法延迟，非HTTP/排队/并发SLA。baseline是上一阶段row-fused快路径，both在其上同时启用长注意力与长扫描。每输入新算current和baseline作为评估参照；首次准备单列在JSON，不纳入分布。115行20组配对；513/615/1024各5次baseline控制、20次候选；长英文和长批次各3次控制、10次候选；512与原多问题短序列各3次配对，图片各2次。交错执行，不将不同样本量混成20组配对。</p>
<div class="scroll"><table><thead><tr><th>输入 / 实际MLP形状</th><th>baseline P50</th><th>both P50</th><th>延迟下降</th><th>both P95</th><th>相对baseline<br>Δ预测 / 门槛</th><th>相对current<br>Δ预测 / 门槛</th><th>长注意力 / 长扫描<br>加速层数</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h3>精度：新改动与累计误差分开判断</h3><p>正式候选{len(current_checked)}份独立对照current，另24次短请求仅核对baseline，共{len(checked)}份输出（不含参照、筛选及带包装的profile调用）。额外短请求JSON中的vs_current字段复用了baseline参照，不计入这里的current误差汇总。相对baseline未通过原门槛的正式场景：<b>{esc(incfailed)}</b>；相对current未通过的正式场景：<b>{esc(curfailed)}</b>。原门槛仍为choice一致且概率/score/noul最大绝对差≤0.01，confidence独立报告，不降低门槛掩盖失败。Δ预测包含score，不能将其直接称为概率百分点。</p><div class="scroll"><table><thead><tr><th>场景</th><th>相对current<br>Δ分类概率</th><th>Δscore</th><th>Δnoul</th><th>Δconfidence</th><th>choice</th></tr></thead><tbody>{''.join(typed)}</tbody></table></div>
<p>“无此字段”表示该请求没有询问对应类型，不能将其解读为该类型已测且误差为零。长英文请求单独测试score；长批次询问kind与programming，没有score。原multi-wide包含score，但实际T=239、B×T=717，本阶段两个长内核均旁路，输出与baseline全部受检数值一致；其此前MLP组合图导致的score偏差继续保留，不能算成本阶段新增偏差，也不能说它已被修复。短请求、512行边界和图片也单独核对本阶段新增差异；这些旁路场景的少量计时差异不作加速结论。图片仍有既有mkldnn BF16不支持、转BLAS的警告，前一阶段日志也存在，本阶段没有修改该视觉路径。没有标注业务数据，不宣称业务准确率无损，不能保证每个新输入都满足0.01。</p>
<h3>单项筛选：收益主要来自哪里</h3><p>每场景每模式3次，提前暖机、轮转顺序。513/615的current参照来自上阶段同一冻结模型；1024在此筛选只对新算baseline，正式阶段已另算current。下表Δ本轮均是相对baseline的预测字段最大差。</p><div class="scroll"><table><thead><tr><th>输入</th><th>baseline</th><th>仅长注意力</th><th>仅长扫描</th><th>两者都扩</th></tr></thead><tbody>{''.join(screenrows)}</tbody></table></div><p>解除资格回退不等于所有长度都更快。长扫描的收益较明显；长注意力的收益较小，1024筛选中未观察到稳定收益。保留这些单项结果，部署时仍需按真实业务输入判断。</p>
<h3>615行组件分解</h3><p>每模式另测3次，开启计时包装；独立于正式分布。注意力/扫描列包含该入口的准备、守卫和执行，baseline列包含原备用路径。其他耗时逐次用总计减两入口后取中位数，因此各列中位数不要求精确相加。其余部分包含NPU投影/MLP、其他CPU算子及输入输出，不能全部归因于某一个NPU瓶颈。</p><div class="scroll"><table><thead><tr><th>模式</th><th>总计ms</th><th>完整注意力ms</th><th>扫描ms</th><th>其他ms</th></tr></thead><tbody>{''.join(profile)}</tbody></table></div>
<h3>边界、资源与线上保护</h3><p>内核单测包括6组注意力形状、5组扫描形状，覆盖513/615/1024/1025/2048、GQA与外层非连续stride、padding mask、float64扫描参照、跨512状态、未知参数、梯度与故障整次回退；2049与缓存/初态/最终态保留原路径。末尾token影响首个query的测试通过，扫描状态跨512测试通过。原生输出先保存在私有缓冲区，失败整次丢弃；故障内核禁用扩展。</p><p>矩阵句柄数在每场景暖机后保持；进程RSS高水位记录在formal.json，并不包含全部NPU驱动内存。注意力工作缓冲动态分配，仍有batch/head/dim/长度保护；扫描受B×head与维度保护。上阶段线性投影B×T≤2048限制独立存在，不能只看序列T就承诺全部batch组合都加速。</p><p>实验进程全部退出。线上容器仍为<code>{esc(a['id'][:12])}</code>，启动时间、配置与{len(a['files'])}个发布文件SHA不变；健康、五次完整HTTP输出与实验前相等。首尾少量HTTP计时不用于判断前后性能相同。本阶段没有部署、提交或推送。</p>
<h3>复现与原始记录</h3><p><a href="../long-sequence-experiment/REPRODUCTION.md">完整复现步骤与适用范围</a> · <a href="../long-sequence-experiment/DESIGN.txt">设计验收</a> · <a href="../long-sequence-experiment/VERIFY.txt">验证命令与结果</a> · <a href="../long-sequence-experiment/logs/native-source.diff">自有C源码改动</a>。冻结推理容器没有GCC，首次容器编译失败记录保留；实际使用RK主机GCC编译新库，再在原镜像加载验证，没有构建镜像或安装软件。</p><p>正式计时后仅补充scaling=None/NaN的原样回退守卫，并重跑聚焦单测；模型使用的有限浮点scale、C计算和两个二进制均未改变，未为该非活动分支重跑整套计时。<a href="../long-sequence-experiment/logs/measured-artifacts.json">计时源码SHA</a>、<a href="../long-sequence-experiment/logs/pre-scale-guard/long_kernels.py">计时适配器源码</a>与<a href="../long-sequence-experiment/logs/scale-guard.diff">最终守卫差异</a>均保留，最终源SHA见artifacts.json。</p><pre># RK主机独立目录内：
bash build.sh
bash run.sh test_kernels.py
bash run.sh screen.py
bash run.sh formal.py --candidate both
# 所有测试退出后，本机仓库根目录顺序执行：
python3 long-sequence-experiment/preserve_service.py after
python3 long-sequence-experiment/capture_artifacts.py
python3 long-sequence-experiment/validate_records.py
python3 long-sequence-experiment/update_report.py</pre><p>{links}</p><p><a href="../long-sequence-experiment/logs/formal.log">正式运行日志</a> · <a href="../long-sequence-experiment/logs/build.log">主机编译日志</a> · <a href="../long-sequence-experiment/logs/build-container-missing-gcc.log">容器缺GCC的首次失败记录</a></p><p><a href="../long-sequence-experiment/service-before.json">service-before</a> · <a href="../long-sequence-experiment/service-after.json">service-after</a> · <a href="../long-sequence-experiment/natural-request.txt">英文长需求探针</a></p><p><b>本阶段仍没有通用200ms结果。</b>解决了已测长序列的两处资格回退，不代表长序列与短请求具有相同计算成本。视频、并发吞吐、其他芯片/模型/SDK、所有长度和batch组合尚未验证。</p></section><div class="box note"><b>以下为前两阶段的历史实测。</b>关于长序列注意力和扫描“尚未扩展”的描述属于当时状态，以第三阶段为准。</div>'''
subprocess.run([sys.executable,str(root/'row-projection-experiment/update_report.py')],check=True,stdout=subprocess.DEVNULL)
path=root/'mlp-length-experiment/report.html';text=path.read_text();text=text.replace('<nav>','<nav><a href="#long-sequence">本轮长序列内核</a>',1).replace('</nav>','</nav>'+section,1)
text=text.replace('d3-edge 多长度 MLP 与投影分块实验','d3-edge 多长度 MLP、投影与长序列内核实验',1)
text=text.replace('多长度 MLP 与线性投影加速记录','多长度推理加速实验记录',1)
text=text.replace('第一阶段由build_report.py生成；第二阶段由row-projection-experiment/update_report.py从原始JSON整合。','前两阶段由各阶段生成器整合；第三阶段由long-sequence-experiment/update_report.py从原始JSON生成。')
path.write_text(text);print('UPDATED',path)
