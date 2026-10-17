# -*- coding: utf-8 -*-
"""Audit frozen records and generate a small evidence-linked report, never execute model."""
from pathlib import Path
import hashlib,html,json,math,statistics
P=Path(__file__).resolve().parent
deploy_path=P.parent/'cpu-tail-deployment/deployment-result.json'
deployment=json.loads(deploy_path.read_text()) if deploy_path.exists() else None
deployed=bool(deployment and deployment.get('passed'))

def read(name):return json.loads((P/name).read_text())
def ms(x):return f'{x*1000:.2f}'
def audit_timing(d,records):
 values=[x['seconds'] for x in records]
 assert len(values)==d['count'] and values==d['seconds']
 assert statistics.median(values)==d['median']
 assert sorted(values)[math.ceil(.95*len(values))-1]==d['p95_nearest_rank']
 assert all(x['passed'] for x in records)

a=read('logs/formal-paired.json');cpu=read('logs/formal-single.json');prof=read('logs/final-profile.json');gc=read('logs/final-graph-comparison.json')
assert a['passed'] and cpu['passed'] and prof['passed'] and gc['passed']
assert len(a['records'])==60 and len(cpu['records'])==100 and len(gc['records'])==60
for n,t in a['timing'].items():audit_timing(t,[x for x in a['records'] if x['variant']==n])
for n,t in gc['timing'].items():audit_timing(t,[x for x in gc['records'] if x['variant']==n])
selected=gc['selected'];graph=selected!='cpu-std';single=read('logs/formal-graph-single.json') if graph else cpu
if graph:prof=read('logs/final-graph-profile.json')
assert single['passed'] and prof['passed'] and len(single['records'])==100
audit_timing(single['timing'],single['records'])
assert len(single['cases'])==7 and len(single['extra'])==12 and len(single['extra_repeats'])==36
for d in single['cases'].values():assert d['passed'] and d['vs_442']['max_delta']<=.01 and d['vs_fp32']['max_delta']<=.01
for d in list(single['extra'].values())+single['extra_repeats']:assert d['passed'] and d['choices_unchanged'] and d['max_delta']<=.01
# Audit actual measured runtime source and ARM libraries, not prose/profile harness edits.
core=['baseline.py','application.py','candidate.py','cpu_backend.py','lut_ops.py','lut_math.c','cpu_attention.c','cpu_attention.py','small_linear.py','rotary.py','rotary.c','residual.py','attention_prepare.py','pipeline.py','pipeline_bridge.cpp','pipeline_profile.cpp','mlp_graph.py','mlp_graph_runtime.cpp','mlp_subgraph.py','mlp_io_codec.h','liblut_math.so','libcpu_attention.so','librotary.so','libpipeline_bridge.so','libpipeline_profile.so','libmlp_graph_runtime.so','libsafe_half_bridge.so','libsafe_scatter_bridge.so']
if graph:core += [f'mlp-{i}-fp16graph.rknn' for i in range(12)]
verified={}
for n in core:
 if n.endswith('.h'):continue # original snapshot enumerates runtime suffixes, not headers
 assert n in single['sources'],n
 digest=hashlib.sha256((P/n).read_bytes()).hexdigest();assert digest==single['sources'][n],n;verified[n]=digest
service=read('service-after.json');assert service['passed'] and not service['remaining_test_containers']
cleanup=read('logs/cleanup.json');assert not cleanup['workers'] and cleanup['socket_absent']
rows=prof['records'];assert len(rows)==10
for r in rows:assert abs(r['total']-sum(x['exclusive'] for x in r['spans'].values()))<1e-7
mean=lambda key:statistics.mean(r[key] for r in rows)
span=lambda key:statistics.mean(r['spans'].get(key,{}).get('exclusive',0) for r in rows)
parts={'SDK调用/同步区间并集':mean('sdk_union'),'桥接转换/分派余项':mean('bridge_remaining'),'CPU GatedDeltaNet扫描':span('scan'),'CPU完整注意力':span('sdpa.custom')+span('sdpa.stock'),'MLP流水准备':span('mlp.pipeline_prepare'),'MLP图API':span('mlp.graph_api'),'归一化/残差融合':sum(span(k) for k in ['norm.native','norm.other','residual_norm.fused']),'卷积/激活':span('causal_conv')+span('activation.lookup'),'注意力准备/RoPE/门控':span('attention.norm_rope')+span('attention.gate')+span('module.other.Qwen3_5TextRotaryEmbedding')}
parts={k:v for k,v in parts.items() if v>0};parts['其余CPU算子/调度/请求处理']=mean('total')-sum(parts.values());assert parts['其余CPU算子/调度/请求处理']>=0
metrics={'original_cases_max_vs_442':max(x['vs_442']['max_delta'] for x in single['cases'].values()),'original_cases_max_vs_fp32_output_reference':max(x['vs_fp32']['max_delta'] for x in single['cases'].values()),'extra_cases_max_vs_442':max(x['max_delta'] for x in single['extra'].values()),'repeat_extra_max_vs_442':max(x['max_delta'] for x in single['extra_repeats'])}
result={'selected':selected,'deployed':deployed,'constraints':{'weights_unchanged':True,'original_12_layers':True,'attention_semantics_unchanged':True,'no_reverse_engineering':True,'threshold':.01},'cpu_paired':{'timing':{k:{t:v for t,v in d.items() if t!='seconds'} for k,d in a['timing'].items()},'paired':a['paired']},'graph_final_comparison':{'timing':{k:{t:v for t,v in d.items() if t!='seconds'} for k,d in gc['timing'].items()},'paired':gc['paired'],'selected':selected},'selected_single100':{k:v for k,v in single['timing'].items() if k!='seconds'},'numerical':metrics,'profile_mean_seconds':mean('total'),'profile_parts_seconds':parts,'verified_runtime_hashes':verified,'service_id':service['container_id'],'service_started_at':service['started_at'],'service_http_before_median':read('service-before.json')['median'],'service_http_after_median':service['median'],'service_preserved':True,'cleanup':cleanup}
result['confidence_audit']=read('../cpu-tail-deployment/confidence-audit.json')
if deployed:result['deployment']=deployment
(P/'experiment-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
summary=f'''RK3576 d3-edge 持续优化结论（独立实验，未部署）
最终候选：{selected}。
100次暖态115-token单问题 system_one：中位数{ms(single['timing']['median'])}ms，P95 {ms(single['timing']['p95_nearest_rank'])}ms，最小{ms(single['timing']['min'])}ms，最大{ms(single['timing']['max'])}ms。不含HTTP。
原442基线与CPU融合版20轮交替：{ms(a['timing']['baseline442']['median'])} -> {ms(a['timing']['cpu-std']['median'])}ms，逐轮改善中位数{a['paired']['cpu-std']['round_reduction_median']*100:.2f}%，20/20更快。
MLP图补充20轮与CPU配对取舍：{selected}。具体分组见experiment-result.json。
输出差异：{json.dumps(metrics,ensure_ascii=False)}；原7场景14问题+额外12场景+36次复测通过，choice不变。
原12层/权重/注意力语义保留；BF16/FP16/FP32执行方式的数值变化经固定.01门槛验证，不代表任务准确率评测。
插桩10次总均值{ms(mean('total'))}ms，各分项见HTML；SDK区间并集不是纯NPU计算。
200ms未达到。已验证当前公开接口下合理的大收益路线，尚无可兑现的明显剩余方案；不等于硬件理论极限或以后绝不可能。
线上三线程容器ID、启动时间、配置与文件哈希未变，5次HTTP输出与实验前完全一致。实验容器/GPU worker已退出。
复现见REPRODUCTION.txt；所有慢样本与失败实验保存在logs。未部署、提交、推送本轮改动。
'''
if deployed:
 summary=summary.replace('（独立实验，未部署）','（2026-10-17实验；2026-10-18已部署）').replace('线上三线程容器ID、启动时间、配置与文件哈希未变，5次HTTP输出与实验前完全一致。实验容器/GPU worker已退出。','上述保留检查是10-17实验阶段记录；10-18新服务已接管8900，旧三线程容器停止保留用于回退。').replace('未部署、提交、推送本轮改动。','当前部署/提交状态见部署报告与Git记录。')
 summary += '\nconfidence补充审计：相对442候选最大0.01002684，旧门槛实际跳过confidence；发布校验另对冻结候选全字段比较。\n'
 summary += '部署LAN HTTP100次：'+str(deployment['http']['timing'])+'\n'
(P/'RESULTS.txt').write_text(summary)
e=html.escape
link=lambda n,label:f'<a href="{e(n)}">{e(label)}</a>'
def table(headers,items):return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{e(str(x))}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in row)+'</tr>' for row in items)+'</tbody></table></div>'
perf=table(['正式协议','版本','中位数 ms','P95 ms','说明'],[['20轮轮换',e(n),ms(t['median']),ms(t['p95_nearest_rank']),'同进程、每次切换后暖机'] for n,t in a['timing'].items()]+[['图补充20轮',e(n),ms(t['median']),ms(t['p95_nearest_rank']),'同一模型、72矩阵句柄、12持久图'] for n,t in gc['timing'].items()]+[['CPU连续100次','cpu-std',ms(cpu['timing']['median']),ms(cpu['timing']['p95_nearest_rank']),'中间候选，独立新进程']]+[['最终连续100次',e(selected),ms(single['timing']['median']),ms(single['timing']['p95_nearest_rank']),'新进程、2次暖机、不剔除样本']])
pt=table(['关键路径分项','插桩均值 ms','占插桩总时长'],[[e(k),ms(v),f'{v/mean("total")*100:.1f}%'] for k,v in sorted(parts.items(),key=lambda x:-x[1])])
nt=table(['输出一致性对照','最大绝对差异','通过'],[[e(k),f'{v:.8f}','≤0.01，choice不变'] for k,v in metrics.items()])
route_rows=[['BF16查表与CPU融合','采用','所有65536种BF16激活域与真实激活布局验证；SiLU、门控、RMSNorm、残差、Q/K norm+RoPE融合。'],['MLP跨投影流水','采用（CPU候选）','直接连接自有SDK半精度缓冲，保留中间BF16舍入，消除张量往返；中间/输出越界整段回退。'],['自有CPU完整注意力','采用','GQA不复制重复头；FP32 SGEMM、稳定softmax、BF16输出，保持非因果mask；最终19场景门槛通过。'],['MLP全12层RKNN子图','采用（最终图候选）','初版F32 I/O 352.44ms vs同轮CPU346.35ms；直接半精度I/O后317.09ms为筛选值，正式结果另列，不能混用。'],['A72编译指令调度','不单独认领收益','早期312.65ms未稳定复现；正式CPU333.84、A72332.00ms，差<1%，优先默认编译。'],['OpenMP/线程/MKLDNN','保留默认3线程','12轮轮换：t3 336.47、t1 344.29、t2 343.37、t4 343.99ms。MKLDNN关闭仅0.05%差异；spin策略未稳定胜出。'],['公开native矩阵布局/转换LUT','未采用','模型筛选lookup435.30、native438.11ms；M115356.13ms，但前后正常对照329.66/361.60ms有漂移，不能认领收益。'],['公开RKNN图双核mask3','未晋级','接口支持combined mask3，但三层实测聚合改善未达5%，不把旧matmul API限制泛化成图API不支持。'],['Mali-G52/Panfrost+CLBlast','未采用','匹配CPU331.88ms，GPU完整注意力537.00ms，数值通过。GPU扫描每层约305ms，六层约1.83s，未做整模型扫描性能/概率声明。'],['原型失败与修复','保留证据','激活布局、softmax/GPU扫描、native布局假设等失败均为自有实现；修复后再验证。标量softmax额外集差异0.0171超门槛，被拒。运行中覆盖脚本的编排错误已修复重跑。']]
routes=table(['路线','取舍','证据/限制'],[[e(x) for x in row] for row in route_rows])
raw=['formal-paired.json','formal-single.json','final-profile.json','final-graph-comparison.json','thread-sweep.json','gpu-screen.json','gpu-unit.json','mlp-graph-screen.json','mlp-half-io-unit.json','mlp-core-probe.json','final-options-screen.json','bridge-screen.json','profile-442.json','cleanup.json']
if graph:raw+=['formal-graph-single.json','final-graph-profile.json']
rawlinks=' · '.join(link('logs/'+x,x) for x in raw)
graphtext='补充交替对照中，图候选稳定快于CPU融合版，最终采用图候选继续100次与剖析。完整注意力仍走原非因果CPU逻辑，只有MLP图使用原权重的FP16运算。' if graph else '补充交替对照未发现满足3%晋级门槛的图候选，最终保留CPU融合版。'
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>d3-edge RK3576 持续优化实测结论</title><style>body{{font-family:system-ui,sans-serif;background:#f4f6fa;color:#172234;margin:0;line-height:1.7}}main{{max-width:1080px;margin:auto;padding:32px 24px 64px}}h1{{line-height:1.25;font-size:32px}}h2{{margin-top:36px}}.badge{{font-size:13px;background:#e0eaff;color:#234c86;padding:6px 10px;border-radius:8px}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:white;padding:20px;flex:1;min-width:180px;border:1px solid #dce3ed;border-radius:12px}}.big{{font-size:30px;font-weight:700;color:#195b89}}table{{width:100%;border-collapse:collapse;background:#fff;font-size:14px}}th,td{{padding:11px;text-align:left;border-bottom:1px solid #dce3ed;vertical-align:top}}th{{background:#eaf0f7}}.scroll{{overflow:auto}}a{{color:#075ba2}}.note{{border-left:4px solid #517ca6;padding:12px 18px;background:#eaf0f7}}code{{overflow-wrap:anywhere}}li{{margin:8px 0}}footer{{font-size:13px;color:#5c687a;margin-top:40px}}</style><main>
<p><span class="badge">2026-10-17 · RK3576 · 独立进程实验 · 未部署</span></p>
<h1>从约442ms继续优化：仍有收益，200ms尚未达到</h1>
<p>本轮保留全部原始12层、权重和注意力语义，不反编译厂商库。最终候选为 <b>{e(selected)}</b>。线上三线程服务保留原版本。</p><p><b>图候选仅加速batch1、115 token的MLP固定形状</b>；其他长度/形状回退到CPU融合路径，其耗时不能套用这里的115-token数字。尚未包装为HTTP服务。</p>
<div class="cards"><div class="card">最终100次中位数<div class="big">{ms(single['timing']['median'])} ms</div>115 token · 单问题 · 暖态方法调用</div><div class="card">最终100次P95<div class="big">{ms(single['timing']['p95_nearest_rank'])} ms</div>nearest-rank · 保留全部样本</div><div class="card">原基线→CPU交替改善<div class="big">{a['paired']['cpu-std']['round_reduction_median']*100:.1f}%</div>20/20轮更快 · 逐轮比例中位数</div></div>
<p class="note">这些是 system_one 方法耗时，包含请求准备/模型/结果处理，不含HTTP和网络，不能当成服务端到端SLA。连续100次与交替对照是不同协议，不能混合相减来宣称精确提速。原线上服务没有换成候选。</p>
<h2>正式测量与最终取舍</h2>{perf}<p>{graphtext} CPU版100次最小{ms(cpu['timing']['min'])}ms、最大{ms(cpu['timing']['max'])}ms；最终候选最小{ms(single['timing']['min'])}ms、最大{ms(single['timing']['max'])}ms。</p>
<p>A72参数早期筛选曾出现312.65ms，但正式交替中未呈现稳定额外收益，因此不归因为指令参数的9%优化。DMC频率等动态状态、温度和所有样本都在原始JSON中；未人为锁频。</p>
<h2>现在主要耗在哪</h2><p>10次插桩总均值 <b>{ms(mean('total'))}ms</b>；插桩前/后各15次无插桩中位数为{ms(prof['before']['median'])}/{ms(prof['after']['median'])}ms。下表按主线程exclusive区间分解，并把重叠SDK区间取并集，合计严格闭合；不是把worker耗时相加。</p>{pt}
<p>SDK调用/同步包含驱动与等待，不是纯设备计算。MLP图API（若启用）包含其输入输出转换和同步，未伪装成细粒度NPU设备时间。分解是插桩均值，不能直接加到100次中位数上。</p>
<p>原442候选中，CPU激活约41.31ms、归一化34.75ms、扫描68.24ms，SDK区间并集均值107.14ms。已用查表、CPU融合、直接缓冲流水等减少大量边界开销；剩下扫描和完整注意力仍是实际计算，12层之间有依赖，无法仅靠增加并行线程消除。</p>
<h2>精度差异：旧概率/score/noul门槛通过，未测任务准确率</h2>{nt}
<p>原7场景覆盖14问题（含不同长度、多问题和图片回退），另12个115-token场景，再重复3轮共36次；全部通过固定≤0.01最大绝对差异门槛，choice不变。所有100次计时也逐次比较。FP32参照是历史<b>FP32输出参考路径</b>，不是全模型纯FP32。扩展集比较冻结442候选；这不是标注数据集上的准确率结论。</p><p><b>2026-10-18勘误：</b>旧比较器跳过confidence，因此上表0.01门槛只涵盖概率/score/noul；补充审计confidence最大绝对差异为0.01002684，不能称所有数值字段均≤0.01。部署HTTP等价性检查包含confidence，与冻结graph-half候选对照。</p>
<p>CPU融合保留BF16关键舍入边界；MLP图用原权重做FP16运算，未量化/裁剪，数值须单独过门槛。未支持形状、训练/梯度、钩子、权重更新和溢出走原路径，定向测试验证回退与资源释放。</p>
<h2>本轮已经验证的路线</h2>{routes}
<p>GPU测试使用私有解包Mesa/Rusticl和公开CLBlast，未安装或替换主机图形驱动。失败原型源代码与日志保留，负面结论只针对本次实现，不能证明GPU硬件理论极限。公开依据：<a href="https://docs.mesa3d.org/drivers/panfrost.html">Mesa Panfrost</a>、<a href="https://docs.mesa3d.org/rusticl.html">Rusticl</a>、<a href="https://github.com/CNugteren/CLBlast/blob/master/doc/api.md">CLBlast API</a>。</p>
<h2>为什么本轮收敛，200ms还差什么</h2><p>当前SDK同步、CPU扫描、完整注意力和桥接余项均已测清，继续删小分配或调整线程没有足够可兑现的收益。即使把约{ms(parts['CPU GatedDeltaNet扫描'])}ms扫描理想化地完全消掉，按这份插桩分解仍未到200ms；这只是按当前分解作量级判断，不是跨算子优化的理论下界。</p>
<p>本轮对有依据的大收益路线完成了CPU融合、桥接/布局、MLP图、公开GPU和调度参数验证，当前没有找到稳定进入200ms的方案，因此停止继续叠加复杂度。这是当前实现与工具范围内的条件性结论，不是“任何工程手段都不可能”。未来若要继续，需新的实质路径，例如可维护的全段原生执行器、能正确表达双向注意力的公开编译器，或更强硬件；均需另行实现和测试，尚无本轮实测收益承诺。</p>
<h2>线上服务核验与清理</h2><p>原容器 <code>{e(service['container_id'][:12])}</code> 启动时间 <code>{e(service['started_at'])}</code> 未变；Config、HostConfig和发布文件SHA256未变，5次HTTP返回与实验前逐值完全一致，health ready / Docker healthy。实验前后HTTP中位数{ms(result['service_http_before_median'])}/{ms(service['median'])}ms为少量健康抽查，不用于性能提升声明。</p><p>测试容器和GPU worker已退出，私有socket不存在，自己的IPC文件已清理；未替换服务、未构建新镜像、未提交/推送本轮实验。</p>
<h2>复现与证据</h2><p>{link('REPRODUCTION.txt','完整复现步骤')} · {link('RESULTS.txt','纯文本结论')} · {link('experiment-result.json','机器可读汇总')} · {link('service-before.json','线上实验前')} · {link('service-after.json','线上实验后')} · {link('PROGRESS.txt','过程与失败记录')}</p><p>复现依赖固定镜像和历史冻结实验目录；不是仅克隆源码即可全新运行的安装包。新机器先按{link('../d3-edge-rk3576-install-guide.html','基础安装指南')}安装，再补齐复现文档列出的工作区。最终CPU运行不需要GPU依赖；可选图比较需12个编译产物。</p><details><summary>原始JSON及算子证据</summary><p>{rawlinks}</p></details><p>验证：定向算子数学/舍入/回退/所有权检查，整模型输出回归，正式交替与100次，插桩分区闭合，核心运行源码和ARM库哈希校验。Python编译和shell语法检查通过；未运行无关仓库全量测试，未评估并发吞吐、冷启动和真实业务任务准确率。</p><footer>由build_report.py从原始记录生成；表格单位均明确。生成器会检查记录完整性、数值门槛、区间闭合和核心哈希；运行环境更换后必须重新验证。</footer></main></html>'''
if deployed:
 from deployment_section import render
 body=body.replace('2026-10-17 · RK3576 · 独立进程实验 · 未部署','2026-10-18 · RK3576 · graph-half已部署；下方保留10-17实验记录')
 body=body.replace('线上三线程服务保留原版本。','新服务已接管8900，旧三线程容器停止保留用于回退。')
 body=body.replace('原线上服务没有换成候选。','本段是方法调用实验口径；实际部署HTTP结果见上方。')
 body=body.replace('尚未包装为HTTP服务。','当前HTTP部署也遵循该回退规则。')
 body=body.replace('<h2>线上服务核验与清理</h2>','<h2>2026-10-17实验阶段保留检查（历史记录）</h2>')
 body=body.replace('未替换服务、未构建新镜像、未提交/推送本轮实验。','该实验阶段未替换服务；2026-10-18已完成下述发布。未构建新镜像。提交/推送状态以Git记录和最终交付为准。')
 body=body.replace('<h1>从约442ms继续优化：仍有收益，200ms尚未达到</h1>', '<h1>graph-half已部署：方法调用318.5ms，服务实测见下</h1>'+render(deployment))
(P/'report.html').write_text(body)
print('AUDIT PASS',selected,'median_ms',ms(single['timing']['median']),'p95_ms',ms(single['timing']['p95_nearest_rank']),'core_hashes',len(verified))
