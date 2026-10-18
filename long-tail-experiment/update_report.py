# -*- coding: utf-8 -*-
from pathlib import Path
import json,html,statistics,subprocess,sys
P=Path(__file__).resolve().parent;root=P.parent
load=lambda n:json.loads((P/n).read_text());esc=lambda x:html.escape(str(x));ms=lambda x:f'{1000*x:.2f}'
f=load('logs/formal.json');s=load('logs/screen.json');p=load('logs/profile-selected.json');original=load('logs/profile.json');replay=load('logs/replay.json');v=load('logs/validation.json');a=load('service-after.json')
assert f['completed'] and v['passed'] and a['passed'];chosen=f['candidate'];target=f['cases']['words-512'];b=target['variants']['baseline']['timing']['median'];t=target['variants'][chosen]['timing']['median']
def table(head,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+x+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
formalrows=[];typed=[];checked=[];summary={}
for key,c in f['cases'].items():
    base,new=c['variants']['baseline'],c['variants'][chosen];rr=new['records'];checked+=rr
    shape=next(iter(rr[0]['mlp']['shapes']),'未记录');inc=max(x['vs_baseline']['all_numeric_max_delta'] for x in rr)
    formalrows.append([esc(key)+'<br>usage='+str(c['input_tokens'])+'<br><code>'+esc(shape)+'</code>',ms(base['timing']['median'])+'<br>n='+str(len(base['records'])),ms(new['timing']['median'])+'<br>n='+str(len(rr)),f'{c["reduction"]*100:+.2f}%',ms(new['timing']['p95_nearest_rank']) if len(rr)>=20 else '小样本，不作尾延迟结论',f'{inc:.8f}',str(rr[0]['mlp_schedule']['parallel_layers'])+' / '+str(rr[0]['long_kernels']['scan']['optimized'])])
    vals=[]
    for field,metric in [('probabilities','probability_max_delta'),('score','score_max_delta'),('noul','noul_max_delta'),('confidence','confidence_max_delta')]:
        present=any(field in x for x in c['current_reference']['answers'].values());vals.append(f'{max(x["vs_current"][metric] for x in rr):.8f}' if present else '—（无此字段）')
    haschoice=any('choice' in x for x in c['current_reference']['answers'].values());choice=('一致' if all(x['vs_current']['choices_unchanged'] for x in rr) else '不一致') if haschoice else '—（无此字段）'
    typed.append([esc(key),*vals,choice,'通过' if new['current_gate_passed'] else '<b>未通过，原有偏差</b>'])
    summary[key]={'baseline':base['timing'],'candidate':new['timing'],'reduction':c['reduction'],'incremental_all_numeric_max_delta':inc,'current_gate_passed':new['current_gate_passed']}
profile_modes=('baseline',chosen)
components=[('attention_entry','完整注意力入口'),('scan_entry','GatedDeltaNet扫描入口'),('mlp','MLP入口（含图等待/收集）'),('projection_rows','其他大型投影入口'),('conv','因果卷积'),('norm_modules','Norm模块'),('small_linear','其他小Linear'),(None,'未包装其余部分')]
profiles=[]
for tag,label in components:
    row=[label]
    for mode in profile_modes:
        records=[r for r in p['records'] if r['mode']==mode];row.append(ms(statistics.median(r['unwrapped'] if tag is None else r['components'].get(tag,{}).get('inclusive',0) for r in records)))
    profiles.append(row)
profiles.append(['总计（独立带包装计时）',*[ms(statistics.median(r['seconds'] for r in p['records'] if r['mode']==mode)) for mode in profile_modes]])
subprofiles=[]
for tag,kind,label in [('scan_c','inclusive','扫描原生计算'),('scan_prepare','inclusive','Q/K归一化、转置准备'),('scan_entry','exclusive','扫描入口的守卫/域检查及其余转换、exp/调度'),('scan_native_wrapper','exclusive','原生包装其余工作（含原输出检查）'),('raw_validate_c','inclusive','新原始输入检查（嵌套子项）'),('finite_c','inclusive','新准备/输出有限数检查（嵌套子项）')]:
    subprofiles.append([label,*[ms(statistics.median(r['components'].get(tag,{}).get(kind,0) for r in p['records'] if r['mode']==mode)) for mode in profile_modes]])
screenrows=[]
for mode,x in s['variants'].items():
    rr=x['records'];screenrows.append([esc(mode),ms(x['timing']['median']),f'{max(r["vs_baseline"]["probability_max_delta"] for r in rr):.8f}','—（无此字段）',f'{max(r["vs_baseline"]["confidence_max_delta"] for r in rr):.8f}'])
pilotrows=[[esc(mode),ms(x['timing']['median']),'通过' if x['passed'] else '未通过'] for mode,x in f['pilot'].items()]
replaytables=[]
for kind,label in [('attention','完整注意力12项'),('scan','扫描29项')]:
    rows=[]
    for name in replay[kind+'_ranking']:
        x=replay[kind][name];rows.append([esc(name),ms(x['sum_layer_medians']),f'{x["max_abs"]:.8f}','通过' if x['passed'] else '未通过',esc(json.dumps(x['config'],ensure_ascii=False))])
    replaytables.append('<details><summary>'+label+'：全部回放筛选结果</summary>'+table(['配置','6层中位数之和ms','激活最大绝对差','张量allclose','参数'],rows)+'</details>')
attbase=replay['attention']['baseline']['sum_layer_medians'];attnew=replay['attention']['q256t3r1']['sum_layer_medians'];scanbase=replay['scan']['baseline']['sum_layer_medians']
initial={tag:statistics.median(r['components'][tag]['inclusive'] for r in original['records']) for tag in ('attention_entry','scan_entry','mlp','projection_rows')}
initialtotal=statistics.median(r['seconds'] for r in original['records'])
newtotal=statistics.median(r['seconds'] for r in p['records'] if r['mode']==chosen);newattention=statistics.median(r['components']['attention_entry']['inclusive'] for r in p['records'] if r['mode']==chosen);withoutattention=statistics.median(r['seconds']-r['components']['attention_entry']['inclusive'] for r in p['records'] if r['mode']==chosen)
summary.update({'candidate':chosen,'candidate_outputs':82,'independent_current_comparisons':58,'formal_incremental_all_numeric_deltas_zero':True,'baseline_original_profile':{'total':initialtotal,**initial},'selected_profile_total':newtotal,'selected_profile_attention':newattention,'production_unchanged':True})
(P/'logs/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
links=' · '.join(f'<a href="../long-tail-experiment/logs/{name}.json">{name}.json</a>' for name in ('profile','replay','selection','unit','extra-unit','screen','formal','profile-selected','artifacts','validation','summary'))
section=f'''<section id="long-tail" class="box"><h2>第四阶段：615行约2.4秒的剩余瓶颈与独立优化</h2>
<p><b>正式20组交错配对，615行中位数从{ms(b)}ms降至{ms(t)}ms，下降{(1-t/b)*100:.2f}%，约{b/t:.2f}倍速度；本轮82份正式及额外候选输出相对上一阶段受检数值完全一致。</b>原模型、12层结构、权重、非因果注意力、扫描状态均保留，未反编译或修改厂商库。线上服务没有替换，以下结果来自无端口的独立进程。</p>
<p>最终候选为<code>{esc(chosen)}</code>：保留原完整注意力和原32行块/64列tile扫描计算，使用自有C/NEON检查并调整现有MLP图的调度。615行相对current累计分类概率差为{max(r['vs_current']['probability_max_delta'] for r in target['variants'][chosen]['records']):.8f}，confidence差为{max(r['vs_current']['confidence_max_delta'] for r in target['variants'][chosen]['records']):.8f}；这是原有累计差异，本轮增量为零。下方保留多问题score的既有失败，不能概括成所有输入精度通过。</p>
<h3>原来的2.4秒主要在哪里</h3><p>首次独立profile为{ms(initialtotal)}ms：完整注意力{ms(initial['attention_entry'])}ms、扫描{ms(initial['scan_entry'])}ms、MLP {ms(initial['mlp'])}ms、其他大型投影{ms(initial['projection_rows'])}ms；剩余包括卷积、Norm、小Linear、输入输出及其他未包装工作。扫描原生计算约302ms，Q/K准备约30ms；另约215ms是入口与包装的其余耗时，包含检查、g/b转换、exp与调度，<b>不能将这215ms全部归因为“校验”</b>。实际收益以下面同进程配对profile和正式计时为准。</p>
<h3>优化后仍耗在哪里：5组独立分项对照</h3><p>单位ms；baseline是上一阶段both，新候选是{esc(chosen)}。本表每模式5次，带计时包装，独立于正式分布。入口间为顶层分解；各列分别取中位数，不强制其和恰等于总中位数。并行图worker耗时重叠，因此只计MLP整体入口，不将各图时间相加。</p>{table(['组件','baseline ms','候选 ms'],profiles)}
<details><summary>扫描嵌套子项：计算与检查分别看</summary><p>exclusive表示扣除已包装子调用的剩余耗时；新检查两项已包含在其父入口/包装内，不再与父项相加。baseline没有调用新C检查，表中的0表示该新函数未调用，不表示baseline没有检查。</p>{table(['子项','baseline ms','候选 ms'],subprofiles)}</details>
<h3>尝试了什么，为什么这样选择</h3><p><b>完整注意力：</b>共享GQA的K/V格式转换、query分块32/64/128/256/615、2/3线程以及除法/倒数归一化。所有query块均访问完整K/V，保持非因果mask，没有把序列截断。真实6层回放中q256/t3/倒数的合计由{ms(attbase)}ms变为{ms(attnew)}ms，激活最大绝对差0.03125；这是激活误差，不是概率差。全模型combined的新增分类概率差约0.00025067。但6次预选combined相对保留原注意力的版本只快不足2%，故最终没有采用这条复杂路径。</p>
<p><b>扫描：</b>比较16/32/64/128行块、32/64/128列tile、2/3线程，以及两种自有NEON直接递推模式。共29项含baseline；没有配置稳定跑赢原32/64扫描。回放baseline合计{ms(scanbase)}ms，仍保留原计算。实际采用的是在原资格保护下用C/NEON检查有限数和g≤0、0≤beta≤1，避免PyTorch检查的临时tensor与多次调度。原始输入、准备结果和输出仍检查，异常仍完整回退。</p>
<p><b>MLP：</b>原615行计划128+128+384不变，每层仍3次图调用、补25行；两个独立句柄分别放到NPU core1与core0，128句柄的两次调用保持串行，与384句柄并行。利用公开API，不改图或权重、不新增图。仅命中[1,615,1024]且计划匹配的输入，其他长度保留原调度。SDK枚举0=AUTO、1=core0、2=core1，不能将旧AUTO模式误写为旧单核模式。输出先放私有缓冲区，全部任务结束后按原行序收集；执行失败整次丢弃并回退。</p>
{table(['初筛模式（各4次）','端到端P50 ms','相对baseline Δ分类概率','Δscore','Δconfidence'],screenrows)}
<p>初筛样本少，不能将单项差值当作精确可相加的收益。另做6次/模式的预选，候选必须同时通过baseline与current门槛；combined比validate_mlp快不足2%时保留原完整注意力。</p>{table(['预选模式（各6次）','P50 ms','双参照门槛'],pilotrows)}
{''.join(replaytables)}<p>回放共41配置、每配置6层×3次；上述合计是“6层中位数之和”，不是全模型P50。原扫描baseline使用旧输出检查，候选使用新检查，因此回放亦包含各自包装检查开销，不能称为纯C算子时间。所有失败或较慢路线保留在原始JSON，不只展示胜出项。</p>
<h3>正式速度：长度、batch和适用范围</h3><p>暖机后交错配对，单位ms，为模型方法耗时，不含HTTP、排队或并发SLA。615行20组，115行10组，其余5/3/2组见下；小样本不作尾延迟结论。usage.input_tokens与实际MLP形状分别列出，batch×T才是MLP行数。只有615行单batch命中此次MLP调度；其他长输入只获得扫描检查优化，短输入保留原路径，不能把小样本波动说成短请求提速。</p>
{table(['输入 / 实际MLP形状','baseline P50','候选 P50','延迟下降','候选P95','相对baseline<br>全部受检数值最大差','MLP并行层 / 长扫描层'],formalrows)}
<h3>精度：本轮零增量，累计偏差仍如实保留</h3><p>九个正式场景58份候选保存完整输出，逐字段重算相对baseline/current差异；额外12个短请求各2次，仅对新算baseline，共82份。615行current新算并与第三阶段冻结记录相等；其余正式场景明确复用该阶段同一模型的current参照。额外短请求的vs_current字段复用baseline，不计入58份独立current比较。所有候选每次真实推理，没有答案缓存。</p>
<p>相对本轮baseline，82份输出的概率、score、noul、confidence受检数值均零增量，choice一致。相对current沿用原门槛：choice一致且概率/score/noul最大绝对差≤0.01，confidence独立报告。score范围0–2，不能把score差当概率百分点；未询问的类型标“无此字段”。没有标注业务集，不能推断业务准确率无损。</p>{table(['输入','相对current<br>Δ分类概率','Δscore','Δnoul','Δconfidence','choice','原门槛'],typed)}
<p><b>multi-wide仍未通过current门槛。</b>它实际T=239、B×T=717，此轮长扫描和615行专用调度均未命中，输出与本轮baseline一致。此前MLP图组合引入的score约0.01745偏差没有被修复，分类概率约0.00971、confidence约0.01773独立保留，不能说成新检查造成，也不能因为新增差为零就忽略。图片仍走原视觉实现，旧mkldnn BF16转BLAS警告见日志。</p>
<h3>往200ms还有没有明显机会</h3><p>优化后615行带包装总计约{ms(newtotal)}ms，其中完整注意力本身约{ms(newattention)}ms，单这一项就超过200ms目标；MLP、其他投影和扫描计算也仍有明显开销。逐次扣除注意力后，其余部分中位数仍约{ms(withoutattention)}ms，说明即使只把这一项消除，也不够进入1秒；这只是当前执行分解，不是未来实现的硬件下限。这轮没有证据支持通过继续调线程或小块参数降到1秒/200ms。615行和此前115行约318ms也不是同一工作量，不能直接套用。</p>
<p>下一步较有针对性的候选是：①重写我们自有完整注意力的CPU融合QK/softmax/PV，减少中间矩阵搬运，或用公开RKNN接口实测分块注意力图；需验证任意mask、非因果、GQA及全K/V归一化，不能只把query分块当作已实现融合。此前115行RKNN完整注意力图曾因数值门槛失败而未采用（见<a href="../d3-edge-development-history.html#R17">R17历史记录</a>），扩到长序列也必须先解决这类误差，不能只扩shape。②按profile细化其他投影的提交、等待、转换，尝试跨独立投影的流水，但必须避免NPU核争用。③若要让多个长度都受益，逐长度/批次评估已有MLP图的双句柄调度，不能直接推广615专用计划。④扫描若继续优化，应针对实际约300ms计算更换实现策略；本轮小块/线程/直接递推已无明显收益。这些仍是待验证方向，不能承诺RKNN能直接接管全部注意力或扫描，更不能据此承诺200ms。</p>
<h3>保护、失败与复现</h3><p>聚焦单测覆盖注意力完整未来K/V、padding mask、GQA/stride、扫描跨512状态、2048边界与2049旁路、非有限数、扫描模型域、未知/缓存参数、原生故障禁用与整次回退；调度假图验证原行序、同句柄不重入、故障时等待另一个worker后丢弃全部结果、恢复AUTO及关闭线程池。它们验证实现边界，不代替真实业务精度。正式场景句柄数暖机后稳定，RSS高水位保存，不代表全部NPU驱动内存。未测试服务并发或多Owner共享。</p>
<p>线上容器仍为<code>{esc(a['id'][:12])}</code>，启动时间、配置及{len(a['files'])}个发布文件SHA均未变，健康、五次完整HTTP输出与实验前一致。自有实验容器已退出。没有替换线上、提交或推送；首尾少量HTTP计时不作性能等价证明。</p>
<p><a href="../long-tail-experiment/REPRODUCTION.md">完整复现、隔离与适用范围</a> · <a href="../long-tail-experiment/VERIFY.txt">验证命令与结果</a> · <a href="../long-tail-experiment/DESIGN.txt">设计验收</a> · <a href="../long-tail-experiment/logs/native-source.diff">自有C源码差异</a>。首次profile实际运行源码存于<a href="../long-tail-experiment/logs/profile-sources/runner.py">profile-sources/runner.py</a>；其余运行源与二进制SHA见artifacts.json。初版编译警告、标准库select模块遮蔽和测试注入别名冲突记录均保留；修复后最终构建无警告、单测通过，不将这些脚本错误混作模型失败。</p>
<pre># RK主机 /opt/decision3-edge-long-tail-20261018 内，逐项运行：
bash build.sh
bash run.sh profile_tail.py
bash run.sh replay.py
bash run.sh choose_candidates.py
bash run.sh test_candidates.py
bash run.sh screen_tail.py
bash run.sh formal_tail.py
bash run.sh profile_selected.py
# 全部退出后，本机仓库根目录：
python3 long-tail-experiment/preserve_service.py after
python3 long-tail-experiment/capture_artifacts.py
python3 long-tail-experiment/validate_records.py
python3 long-tail-experiment/update_report.py</pre>
<p>{links}</p><p><a href="../long-tail-experiment/logs/formal.log">正式日志</a> · <a href="../long-tail-experiment/logs/profile-selected.log">最终分项日志</a> · <a href="../long-tail-experiment/logs/unit.log">单测日志</a> · <a href="../long-tail-experiment/logs/build.log">最终编译日志</a> · <a href="../long-tail-experiment/service-before.json">service-before</a> · <a href="../long-tail-experiment/service-after.json">service-after</a></p></section>
<div class="box note"><b>以下为前三阶段历史记录。</b>第三阶段615行约2.4秒是本轮起点；本轮结论以第四阶段为准。历史筛选、累计精度偏差与尚未扩展范围按原记录保留。</div>'''
subprocess.run([sys.executable,str(root/'long-sequence-experiment/update_report.py')],check=True,stdout=subprocess.DEVNULL)
path=root/'mlp-length-experiment/report.html';text=path.read_text();text=text.replace('<nav>','<nav><a href="#long-tail">615行剩余瓶颈优化</a>',1).replace('</nav>','</nav>'+section,1)
text=text.replace('d3-edge 多长度 MLP、投影与长序列内核实验','d3-edge 多长度推理与615行剩余瓶颈实验',1)
text=text.replace('前两阶段由各阶段生成器整合；第三阶段由long-sequence-experiment/update_report.py从原始JSON生成。','各历史阶段由对应生成器整合；第四阶段由long-tail-experiment/update_report.py从原始JSON生成。')
path.write_text(text);print('UPDATED',path)
