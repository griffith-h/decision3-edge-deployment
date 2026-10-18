# -*- coding: utf-8 -*-
"""Build a self-contained report from completed raw experiment records."""
from pathlib import Path
import json,html,statistics,ast,datetime
P=Path(__file__).resolve().parent
load=lambda n:json.loads((P/n).read_text())
esc=lambda s:html.escape(str(s))
ms=lambda x:f'{x*1000:.2f}'
pct=lambda x:f'{x*100:+.2f}%'
formal=load('logs/planned-formal.json');long=load('logs/long-screen.json');screens=[('115行分块','screen.json','tile115'),('128/384粗桶','bucket-screen.json','bucket'),('按成本组合','planned-screen.json','planned')]
assert formal['passed'] and long['completed']
for name in ['row-unit','bucket-unit','planned-unit','bucket-primitive']:
 assert load('logs/'+name+'.json')['passed'],name
before,after=load('service-before.json'),load('service-after.json');assert after['passed']
for k in ['id','started_at','config_hash','hostconfig_hash','files']:assert before[k]==after[k],k
assert all(v==before['responses'][0] for v in after['responses'])
rows=[];numeric=[];per=[]
for key,c in formal['cases'].items():
 a,b=c['variants']['fixed'],c['variants']['planned'];ra,rb=a['records'],b['records'];assert len(ra)==len(rb)==20
 numeric+=rb;gain=1-b['timing']['median']/a['timing']['median'];wins=sum(y['seconds']<x['seconds'] for x,y in zip(ra,rb));paired=statistics.median(1-y['seconds']/x['seconds'] for x,y in zip(ra,rb));shape=next(iter(rb[0]['shapes']));plan=' / '.join(k.replace(',','+') for k in rb[0].get('plans',{})) or '原115行图'
 if key=='single':verdict='保留原115路径'
 elif not all(x['passed'] for x in rb):verdict='精度未通过'
 elif gain>.02 and wins>=16:verdict='有收益；需业务回归'
 elif gain<-.02:verdict='更慢；保留原路径'
 else:verdict='收益小；暂不推广'
 per.append({'id':key,'input_tokens':c['input_tokens'],'shape':shape,'baseline_ms':a['timing']['median']*1000,'candidate_ms':b['timing']['median']*1000,'reduction':gain,'paired_median_reduction':paired,'faster_pairs':wins,'verdict':verdict})
 rows.append(f'<tr><td>{esc(key)}</td><td>{c["input_tokens"]}</td><td><code>{esc(shape)}</code><br>{esc(plan)}</td><td>{rb[0]["graph_calls"]}</td><td>{ms(a["timing"]["median"])}</td><td>{ms(b["timing"]["median"])}</td><td>{pct(gain)}<br>{wins}/20组更快</td><td>{ms(a["timing"]["p95_nearest_rank"])} /<br>{ms(b["timing"]["p95_nearest_rank"])}</td><td>{max(x["max_delta"] for x in rb):.8f}<br>{max(x["confidence_max_delta"] for x in rb):.8f}</td><td>{esc(verdict)}</td></tr>')
for c in formal['extra']:
 assert c['passed']
 if c['mode']=='planned':numeric.append(c)
for _,file,mode in screens:
 d=load('logs/'+file);assert d['passed']
 for c in d['cases'].values():numeric+=c['variants'][mode]['records']
for c in long['cases'].values():numeric+=c['variants']['planned']['records']+[c['first_check']]
maxp=max(v['max_delta'] for v in numeric);maxc=max(v['confidence_max_delta'] for v in numeric);maxa=max(v['all_numeric_max_delta'] for v in numeric);choices=all(v['choices_unchanged'] for v in numeric);normal_errors=sum(len(v.get('errors',[])) for v in numeric);fallbacks=sum(v.get('range_fallback',0) for v in numeric)
phase_rows=[]
for key in load('logs/planned-screen.json')['cases']:
 cells=[]
 for _,file,mode in screens:
  c=load('logs/'+file)['cases'][key];a,b=c['variants']['fixed']['timing']['median'],c['variants'][mode]['timing']['median'];n=c['variants'][mode]['timing']['count'];cells.append(f'<td>{ms(a)} → {ms(b)}<br>{pct(1-b/a)} · 各{n}次</td>')
 phase_rows.append('<tr><td>'+esc(key)+'</td>'+''.join(cells)+'</tr>')
long_rows=[]
for key,c in long['cases'].items():
 a,b=c['variants']['fixed'],c['variants']['planned'];rb=b['records'];long_rows.append(f'<tr><td>{c["input_tokens"]}</td><td>{esc(next(iter(rb[0]["shapes"])))}</td><td>{ms(a["timing"]["median"])}</td><td>{ms(b["timing"]["median"])}</td><td>{pct(c["reduction"])}</td><td>{rb[0]["graph_calls"]}</td><td>{max(v["max_delta"] for v in rb):.8f}<br>{max(v["confidence_max_delta"] for v in rb):.8f}</td><td>{"通过" if b["passed"] else "失败"}</td><td>{ms(c["first_fixed_seconds"])} / {ms(c["first_planned_seconds"])}</td></tr>')
profile=load('logs/planned-profile.json');profilerows=[]
for key,modes in profile.items():
 for mode,d in modes.items():
  rest=statistics.median(v['seconds']-v['mlp_seconds'] for v in d['records']);profilerows.append(f'<tr><td>{esc(key)}</td><td>{esc(mode)}</td><td>{ms(d["total"]["median"])}</td><td>{ms(d["mlp"]["median"])}</td><td>{ms(rest)}</td></tr>')
passed=all(v['passed'] for v in numeric) and long['passed']
summary={'formal':per,'max_probability_score_noul_delta':maxp,'max_confidence_delta':maxc,'max_all_numeric_delta':maxa,'choices_unchanged':choices,'checked_candidate_outputs':len(numeric),'all_passed':passed,'runtime_errors':normal_errors,'range_fallbacks':fallbacks,'long_passed':long['passed'],'service_unchanged':True}
(P/'logs/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
links=['screen.json','bucket-screen.json','planned-screen.json','planned-formal.json','long-screen.json','planned-profile.json','bucket-primitive.json','row-unit.json','bucket-unit.json','planned-unit.json','baseline-http-compat.json','local-artifacts.json','board-artifacts.json','record-validation.json','summary.json']
linklist=' · '.join(f'<a href="logs/{name}">{name}</a>' for name in links)
good=[v for v in per if v['verdict'].startswith('有收益')]
short_note=(f"20组正式对照中，{len(good)}个场景达到本轮筛选规则，延迟下降{min(v['reduction'] for v in good)*100:.1f}%–{max(v['reduction'] for v in good)*100:.1f}%；其余场景收益较小或保留原路径。" if good else '本轮正式对照尚未发现达到筛选规则的收益。')
long_candidates=[]
for c in long['cases'].values():
 shape=ast.literal_eval(next(iter(c['variants']['planned']['records'][0]['shapes'])))
 if shape[0]*shape[1]>512:long_candidates.append(c)
long_note=''
if long_candidates:
 gains=[v['reduction'] for v in long_candidates];long_note=f"512行以上另补测{len(gains)}个场景，观察到延迟下降{min(gains)*100:.1f}%–{max(gains)*100:.1f}%，每模式仅2次，需更多业务样本复核。"
 if not long['passed']:long_note+='其中存在概率门槛或选择一致性失败，不能推广该范围。'
report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>d3-edge 多长度 MLP 图实验</title><style>
:root{{color-scheme:light;--ink:#183047;--muted:#526779;--line:#d5e0e8;--blue:#136489}}*{{box-sizing:border-box}}body{{margin:0;background:#eef3f6;color:var(--ink);font:16px/1.75 system-ui,-apple-system,"PingFang SC",sans-serif}}main{{max-width:1320px;margin:auto;padding:36px 28px 80px}}h1{{font-size:32px;line-height:1.3}}h2{{font-size:23px;margin-top:38px}}h3{{font-size:19px}}a{{color:var(--blue);overflow-wrap:anywhere}}p{{max-width:1100px}}.box{{background:white;padding:22px;border:1px solid var(--line);border-radius:12px;margin:20px 0}}.note{{border-left:5px solid #d49b37}}.muted,small{{color:var(--muted)}}.scroll{{overflow-x:auto;background:white;border:1px solid var(--line);border-radius:8px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}}th{{background:#e3edf3;white-space:nowrap}}code,pre{{font-family:ui-monospace,monospace;font-size:13px}}pre{{background:#172f43;color:#edf6fd;padding:18px;overflow:auto;border-radius:8px}}nav{{display:flex;gap:14px;flex-wrap:wrap}}li{{margin:8px 0}}@media(max-width:700px){{main{{padding:22px 15px}}h1{{font-size:27px}}.box{{padding:16px}}}}
</style><main><p class="muted">RK3576 · 2026-10-18 · 独立实验进程 · 线上 graph-half v1 保留</p><h1>让不同长度的请求复用 MLP 图</h1><nav><a href="#conclusion">结论</a><a href="#formal">正式对照</a><a href="#accuracy">精度差异</a><a href="#boundary">512行边界</a><a href="#routes">三条路线</a><a href="#profile">耗时分解</a><a href="#reproduce">复现</a></nav>
<section id="conclusion" class="box"><h2>结论与使用范围</h2><p><b>{esc(short_note)}</b> {esc(long_note)}</p><p><b>已实现不同长度的 MLP 图复用，收益随长度变化，不能全量强制启用。</b>原115行图之外新增128/384行图，再按实测成本组合分块。MLP收到的 <code>B×T</code> 行展平后独立计算，只有最后一块在MLP内部补零；序列、attention mask、位置和GatedDeltaNet扫描不变。</p><p>表中“有收益”表示本组20次配对的中位延迟下降超过2%，且至少16组更快，是本轮筛选依据，不是所有业务输入的性能保证。原115-token路径保留。其他形状原来已有RKNN矩阵加速和CPU融合；本轮相对它继续比较，不能按纯CPU基线计算加速倍数。</p><p>权重、12层结构与计算公式不变；FP16图融合改变中间舍入，故输出有数值差异。测试采用原始模型和公开SDK，没有裁剪、反编译或替换厂商库。独立测试中观测到实验进程约4.9 GiB、现有服务约3.7 GiB内存；前者持有额外图集，两种对照路径共享相同实验图集。该读数不是峰值。尚未替换线上服务，本轮没有提交或推送代码。</p></section>
<section id="formal"><h2>正式对照：每个长度20组交替计时</h2><p>同一常驻模型、三线程、CPU4–7、相同输入，先暖机再交替运行fixed与planned。单位ms。fixed只在 <code>[1,115,1024]</code> 用原MLP图；其他形状走原执行路径。planned在同一模型上改用组合图。input_tokens与MLP行数含义不同，多问题请求尤其不能混用。words-* 是重复alpha构造的长度探针，不代表真实业务输入分布。</p><div class="scroll"><table><thead><tr><th>场景</th><th>input_tokens</th><th>实际MLP形状 / 选图</th><th>图调用/请求</th><th>原路径P50</th><th>组合图P50</th><th>延迟下降 / 配对</th><th>原/组合P95</th><th>最大Δ概率等<br>最大Δconfidence</th><th>判定</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div><p class="muted">延迟下降=1−候选中位数/基线中位数；负数为更慢。P95采用nearest-rank，20个样本仅作描述。115路径执行相同图，小幅差异属于计时波动。每次正式输出均核对；另有12个115-token请求各3组回归。首次形状准备开销保留在筛选记录，不混入暖机分布。这是模型方法延迟，不含HTTP/排队，不能承诺并发SLA。</p></section>
<section id="accuracy" class="box note"><h2>精度与回退检查</h2><p>共核对<b>{len(numeric)}</b>份候选输出（不含作为参照的fixed调用）。概率/score/noul最大绝对差异 <b>{maxp:.8f}</b>；confidence最大绝对差异 <b>{maxc:.8f}</b>；答案中全部受检数值字段最大差异 <b>{maxa:.8f}</b>；choice一致：<b>{'是' if choices else '否'}</b>；概率等门槛全通过：<b>{'是' if passed else '否'}</b>。</p><p><b>0.01门槛只适用于概率/score/noul，confidence另行报告。</b>例如0.014表示confidence差约1.4个百分点，不是业务准确率下降1.4%。比较范围是answers预测字段，排除legend等元数据；0.01是离线样本验收门槛，不是每个新输入的运行保证。没有标注数据集，不能据此宣称业务准确率无损。若业务依赖confidence阈值，也必须单独回归。</p><p>正常计时调用中的运行错误={normal_errors}，FP16范围回退={fallbacks}。边界单测验证了非连续输入、批次展平、权重变化/训练/梯度/钩子保护、第二块失败整层回退、故障层禁用和构造失败清理。硬件抽查0/5/11层：补零输出为零、行置换结果逐行一致、真实115行前缀对比原图完全一致，NaN/Inf/溢出输入拒绝后仍可正常运行。</p><p>代码支持最多2048个MLP行并保持原输出形状；超过上限或不符合CPU/BF16等条件走原路径，无截断。1–2048是实现保护范围，不代表每个长度已做完整模型精度/性能验证。</p></section>
<section id="boundary"><h2>512行边界与长输入补测</h2><p>现有矩阵乘适配器的eligible上限是512个MLP行；这是当前适配器边界，不能当作RK3576硬件上限。新MLP图分块可跨过该边界，但其他投影仍沿原路径。以下各模式暖机一次、交替计时两次，独立于20组正式对照；不能将这两次结果当作P95或稳定SLA。</p><div class="scroll"><table><thead><tr><th>input_tokens</th><th>实际MLP形状</th><th>原路径中位ms</th><th>组合图中位ms</th><th>延迟下降</th><th>图调用</th><th>Δ概率等<br>Δconfidence</th><th>门槛</th><th>首次原/候选ms</th></tr></thead><tbody>{''.join(long_rows)}</tbody></table></div></section>
<section id="routes"><h2>三条路线的筛选记录</h2><p>每个单元格均为该阶段自己的配对基线→候选，不跨阶段相除。文本每模式4次，粗桶与组合图图片每模式2次；首轮115分块图片每模式4次。观察到变慢的长度也完整保留。</p><div class="scroll"><table><thead><tr><th>场景</th><th>复用115图拆块</th><th>128/384粗桶</th><th>按实测成本组合</th></tr></thead><tbody>{''.join(phase_rows)}</tbody></table></div><p>组合版用0/5/11层的零缓冲区测各图15次暖机调用，汇总中位耗时，再为1–2048行预计算成本最小的静态图组合。这里只缓存执行计划，没有输入或答案缓存。校准成本是估计，实际收益仍由完整请求验证；原生RKNN动态profile的编译与性能未在本轮测试。</p></section>
<section id="profile"><h2>MLP之外还剩多少开销</h2><p>独立的5次带计时包装调用，记录12层MLP耗时之和及其他开销。此表用于分解，不能与正式对照混作同一计时分布。其他开销逐次用总耗时−MLP耗时计算后取中位数，含注意力、扫描、其余投影及输入/输出处理。</p><div class="scroll"><table><thead><tr><th>场景</th><th>路径</th><th>总计ms</th><th>12层MLP ms</th><th>其他开销ms</th></tr></thead><tbody>{''.join(profilerows)}</tbody></table></div><p>本轮没有验证出通用200ms方案；长输入只加速MLP仍保留其他模块的开销。下一步可单独验证其他线性投影按行分块，或调整公开矩阵接口的适配范围，尝试消除超过512行的回退；本轮未实现或测量这一步，不能承诺其耗时或数值一致性。这里只对表内实际输入给出结论，不外推到视频、多并发、其他RK芯片或更换权重的模型。</p></section>
<section id="reproduce" class="box"><h2>复现与线上保护</h2><p>完整命令、固定依赖、编译方法和边界见 <a href="REPRODUCTION.md">REPRODUCTION.md</a>；设计验收见 <a href="DESIGN.txt">DESIGN.txt</a>；验证命令与结果见 <a href="VERIFY.txt">VERIFY.txt</a>。本机x86 Docker转换24个非量化图，原115图只读复用；板端只启动无端口的独立测试容器。新代码位于本目录，冻结发布目录全程只读挂载。</p><pre>python3 mlp-length-experiment/preserve_service.py before
bash mlp-length-experiment/compile.sh
# 同步到独立实验目录、编译公开API包装，步骤见REPRODUCTION.md
# 在板端依次运行，每条结束再运行下一条：
bash run.sh test_planned.py
bash run.sh probe_buckets.py
bash run.sh screen_planned.py
bash run.sh formal_planned.py
bash run.sh profile_planned.py
bash run.sh screen_long.py
# 停止本轮测试进程后，在本机验证现有服务：
python3 mlp-length-experiment/preserve_service.py after</pre><p>实测线上保护通过：容器<code>{esc(before['id'][:12])}</code>，启动时间、Config/HostConfig哈希、{len(before['files'])}个发布文件哈希不变；健康状态正常，五次HTTP响应与实验前完整相等，测试容器已退出。直接模型方法对线上HTTP的单一样例兼容性差异单独记录，不声称跨调用封装逐字节等价。此校验确认身份、文件与输出，少量HTTP样例不用于宣称前后延迟相同。</p><p>原始记录：{linklist}</p><p><a href="service-before.json">service-before.json</a> · <a href="service-after.json">service-after.json</a> · <a href="../d3-edge-development-history.html">此前完整开发记录</a></p></section><p class="muted">所有数据来自本轮板端实测。报告由build_report.py从原始JSON生成。</p></main></html>'''
(P/'report.html').write_text(report)
print(json.dumps(summary,ensure_ascii=False,indent=2))
