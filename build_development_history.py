# -*- coding: utf-8 -*-
"""Build the single chronological research record from narrative and retained evidence.
No model, service, Docker, SSH, or network calls. --check is a documentation audit.
"""
import argparse, hashlib, html, json, re, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'d3-edge-development-history.html'
SELF={'build_development_history.py','d3-edge-development-history.html','history-verification.json'}
GROUPS=[('root','最初部署与服务状态'),('rkllm-experiment','RKLLM转换与真机兼容性'),('rknn-matmul-experiment','RKNN MatMul初版'),('rkllm-d3-support','RKLLM公开适配研究'),('d3-edge-rknn-handoff','独立安装文档交接'),('latency-experiment','CPU算子与线程'),('optimized-deployment','三线程正式部署'),('bottleneck-analysis','859ms瓶颈与公开核心接口'),('bridge-parallel-experiment','桥接融合与投影并行'),('scan-optimization-experiment','CPU GatedDeltaNet扫描'),('remainder-optimization-experiment','FP16输出及范围保护'),('projection-compile-experiment','投影合并/负载/CPU编译'),('remaining-500ms-experiment','卷积布局/转换/扫描前处理'),('attention-chunk-experiment','完整注意力图与块扫描'),('cpu-tail-experiment','442ms以后所有分支'),('cpu-tail-deployment','最终graph-half部署')]
GROUP_NAMES=dict(GROUPS)
references=set()
def read(n):return json.loads((ROOT/n).read_text())
def e(v):return html.escape(str(v),quote=True)
def link(n,label=None):
 assert (ROOT/n).is_file(),n
 references.add(n)
 return f'<a href="{e(n)}">{e(label or n)}</a>'
def ms(v):return f'{v*1000:.2f}'
def s(v):return f'{v:.3f}'
def scalar(v):
 if isinstance(v,bool):return 'true' if v else 'false'
 if isinstance(v,float):return f'{v:.9g}'
 return str(v)
def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+e(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def slim_timing(d):return {k:v for k,v in d.items() if k not in ('seconds','records','requests','samples') and isinstance(v,(str,int,float,bool))}
def screen(n):
 d=read(n);rows=[]
 for name,v in d['variants'].items():
  t=v.get('timing',{});rows.append([e(name),ms(t['median']) if 'median' in t else '未汇总',e(t.get('count','—')),e(v.get('passed','未给定')),e(json.dumps(v.get('config',d.get('configs',{}).get(name,{})),ensure_ascii=False))])
 return '<p class="small">'+link(n)+'；本表为该轮筛选协议，不能与其他轮直接相减。passed仅代表文件原比较器口径，不包含额外未测字段。</p>'+table(['原始版本名','中位数 ms','次数','原始passed','配置（若记录）'],rows)

sections=[]
def phase(code,title,question,change,result,decision,files,extra=''):
 evidence=' · '.join(link(n) for n in files)
 sections.append((code,title,f'<section id="{code}"><div class="eyebrow">{code} / 按实施依赖排序</div><h2>{e(title)}</h2><p><b>要解决的问题：</b>{question}</p><p><b>这版做了什么：</b>{change}</p><p><b>测到了什么：</b>{result}</p><p><b>最终取舍：</b>{decision}</p>{extra}<p class="sources"><b>原始证据 / 源码 / 复现：</b>{evidence}</p></section>'))

smoke=read('smoke-results.json');opt=read('optimization-results.json');rk=read('rknn-matmul-experiment/experiment-result.json');llm=read('rkllm-experiment/experiment-result.json');latest=read('cpu-tail-deployment/deployment-result.json')
phase('R01','初次连板：官方CPU BF16服务','先确认局域网RK3576能加载并提供d3-edge决策接口。','在NanoPi R76S / RK3576 ARM64 Linux使用固定镜像与官方d3_server，模型快照固定为20c745653400d191ecacd99c4eafc0c5900dc15e；完整文件哈希校验，官方12层，CPU BF16执行，8899提供原始API。','中文多问题请求47.217s、另一中文请求43.232s，英文三问47.845s，小图片72.409s；115-token单问重启抽查15.553s。这些是不同请求/时点的单次数据，不能混为统一基线。','API打通，速度不足。ARM CPU与模型包附带CPU参考并非逐位相同，初始0.001一致性目标未满足，不能把初版当成绝对数值真值。',['deployment.json','smoke-results.json','golden-results.json','restart-results.json'],table(['场景','token','问题数','HTTP秒（单次）'],[[e(x.get('state','')[:28]),x['response']['usage']['input_tokens'],len(x['response']['answers']),s(x['seconds'])] for x in smoke['requests']]+[[e(x['kind']),x['response']['usage']['input_tokens'],len(x['response']['answers']),s(x['seconds'])] for x in read('golden-results.json')]))
phase('R02','CPU FP32 Linear/Conv，保留BF16边界','RK3576上BF16 Linear/Conv路径太慢。','为符合条件的Linear/Conv缓存FP32权重/偏置，FP32计算后回到原BF16激活边界；不改检查点、不剪层，视觉/文本都按原模块执行。','中文47.217→17.990s、43.232→16.970s；英文47.845→15.699s；图片72.409→20.939s。115-token单问7.220s，线上抽查7.170s，重启7.367s。这四类观测约2.55–3.46倍；115-token抽查另列，不把不同请求混并为统一倍数。','部署为后续CPU参考与回退；样本概率/score/noul最大差0.0056557191，choice一致。数值近似，不声称无损或任务准确率等价。',['cpu_optimized_server.py','optimization-results.json','optimized-single-result.json','optimized-production-result.json','optimized-restart-result.json'],table(['场景','原BF16秒','FP32适配秒','文件记录最大差'],[[e(x['kind']),s(x['baseline_seconds']),s(x['seconds']),scalar(x['max_probability_or_score_delta'])] for x in opt]))
phase('R03','官方RKLLM：原12层W8A8 → 18槽位FP16兼容原型','官方有Qwen3.5路径，能否转换整个文本编码器上NPU？','在ARM64 Mac的linux/amd64 Docker仿真中使用官方RKLLM1.3.1。原12层W8A8导出成功，但真机初始化缺少blk.2.ssm_conv1d.weight。诊断到Runtime按周期推断层类型，与发布模型不规则层排列冲突。临时加入6个恒等槽位，将导出布局扩成18层；CPU上因果/非因果模式的隐状态对照均为0差。此兼容原型只用于诊断，不是最终模型，也不是额外裁剪。','原W8A8转换约457.96s；padding非量化v1/v2导出约35.24/34.87s，这些是转换耗时，不是推理速度。v2把metadata hook移到头部/键值计数序列化前，实际读回causal=false。真机10次最后隐状态请求完成，单问题2.705–3.471s（非HTTP），NPU有负载。','语义失败，未部署。最大概率差0.752699，4个choice只匹配2个；NPU隐状态与因果CPU余弦0.99999976，与原非因果CPU仅0.77062291。写入causal=false不代表Runtime执行正确；重新量化或重命名模型不能修复此处语义。',['rkllm-experiment/experiment-result.json','rkllm-experiment/logs/convert-report.json','rkllm-experiment/logs/convert-padded-noncausal-report.json','rkllm-experiment/logs/convert-padded-noncausal-v2-report.json','rkllm-experiment/logs/exported-attention-metadata.json','rkllm-experiment/logs/hidden-diagnosis.json','rkllm-experiment/prepare_padding.py'])
phase('R04','RKNN MatMul混合执行：只把矩阵乘交给NPU','放弃有语义偏差的整模型Runtime，能否保留CPU非因果注意力？','使用公开RKNN MatMul API，78个文本Linear、78个持久句柄，FP16输入/权重与FP32输出，回到BF16边界；CPU保留注意力、DeltaNet扫描、视觉塔和readout。动态行数上限512，超限原CPU回退。','同进程3次中位数：115-token单问CPU5.339→RKNN2.442s；中文多问题15.091→5.504s；另一中文13.606→5.224s；英文12.928→5.230s；图片17.118→11.971s。LAN HTTP单问CPU抽查6.686s，RKNN三次中位数2.822s。','合格并启动独立8900，保留8899 CPU。5用例12问题最大差0.00283136，choice一致。16项矩阵微测与动态/512边界/释放回退通过；不能把矩阵微测速度当整模型提升。',['rknn-matmul-experiment/experiment-result.json','rknn-matmul-experiment/logs/benchmark-all.json','rknn-matmul-experiment/logs/api-verification-before-worker.json','rknn-matmul-experiment/logs/api-verification-fixed-worker.json','rknn-matmul-experiment/logs/thread-diagnosis.json','rknn-matmul-experiment/rknn_linear.py'],table(['用例','CPU模型内秒','RKNN模型内秒','最大概率/分数差'],[[e(k),s(v['cpu_seconds']),s(v['rknn_seconds']),scalar(v['max_probability_or_score_delta'])] for k,v in rk['model_benchmarks']['cases'].items()]))
phase('R05','早期HTTP工作线程诊断与用户关停','NPU核心计算之外，服务线程会影响真实HTTP速度；同时必须遵守关停要求。','诊断初始化主线程与请求工作线程的Torch/OpenMP条件差异，把模型执行固定在初始化主线程，后台HTTP读取仍可并发。随后按用户要求停止两个服务并设restart=no，核对停止状态与端口关闭；重新启用属于R08。','SDK单矩阵core mask0/1/2可用，matmul组合mask3被拒，不能由此推断所有图API都拒绝mask3。重启抽查验证接口/422和历史输出，未把重启抽查称新成对性能试验。','保持客户端请求格式不变；后续两个正式服务都沿用“主线程串行模型+HTTP线程”的所有权。关停和恢复是独立运维事件，不是新的算法提速。',['rknn-matmul-experiment/diagnose_thread.py','rknn-matmul-experiment/npu_server.py','shutdown-verification.json'])
phase('R06','尝试修复RKLLM：公开导出适配与独立C++策略','能否直接修改官方开源仓库，让Qwen3.5优化路径支持原12层非因果注意力？','核对当时官方固定提交f7390530，原加载器已经选Qwen3_5DecoderLayer映射。实现原配置哈希/层序列校验、metadata写入、完整/线性注意力分别标识；编写可移植C++层选择、非因果padding mask、完整K/V就绪及历史隔离策略。','6个Python测试通过，官方Writer真实写入/读回通过；C++策略在macOS和ARM板CPU单测通过。公开Runtime是预编译librkllmrt.so，转换核心也是编译模块；公开接口未提供所需self-attention mask或逐层调度入口。','原生修复受阻：这些组件未被官方执行器加载，没有新的模型推理或提速成绩。没有进行反编译，也没有声称只改metadata就能生效。交叉注意力mask和普通上游llama.cpp不能代替本版本官方NPU执行器。',['rkllm-d3-support/result.json','rkllm-d3-support/native-integration-requirements.json','rkllm-d3-support/patches/d3_export_adapter.py','rkllm-d3-support/patches/d3_attention_policy.hpp','rkllm-d3-support/logs/writer-patch-readback.json','rkllm-d3-support/logs/sdk-public-api.json'])
phase('R07','交接文档与独立安装草案','对方要在相同RK机器上从头安装，不能依赖我们现成容器。','整理自包含HTML基础安装文档、固定版本、SDK/模型下载、源码初始化、调用及复现命令；d3-edge-rknn-handoff保存安装草案和有限管理测试。用户随后明确只给文档，不要求代建镜像。','适配核心已有板端验证；独立从零安装整条路径未在空白机器完整跑通。命令语法、内嵌源码还原和文档页面检查有记录。','文档交接完成，但不把静态检查当新机器安装成功。最终318ms升级还依赖后续源码、ARM库、12个图，基础单HTML不等于已包含这些新产物。',['d3-edge-rk3576-install-guide.html','install-guide-verification.json','d3-edge-rknn-handoff/manage.py','d3-edge-rknn-handoff/test_manage.py','README.md'])
phase('R08','从约2秒剖析：向量卷积、简化扫描、文本FP32 SDPA','不改模型、不反编译，原RKNN请求约2秒耗在哪？','10-17重新启用时只恢复RKNN8900，CPU8899保持关闭；这是重启抽查，不是新性能成对试验。随后首轮函数/模块墙钟剖析总约2.267s，因果卷积约1.207s最突出。用FP32累加的等价depthwise卷积保留激活前BF16舍入；删去零初态及不返回最终状态的无用扫描工作；仅文本Qwen3.5注意力使用FP32 SDPA。','首轮HTTP20次中位885.72ms、P95915.26ms；新进程40次中位988.79ms、P951018.68ms、最慢1829.45ms，37/40低于1秒。GC约1ms，分配器参数和仅改OPENBLAS_NUM_THREADS=1均未解决慢峰。','首轮“部分样本进1秒”成立，“稳定1秒内”不成立。全局FP32 SDPA包含视觉时图片差0.01886348超限，收缩到文本后保留视觉原路径。完整Kineto跟踪处理过重而中止，保留轻量墙钟资料。',['service-activation-verification.json','latency-experiment/result.json','latency-experiment/logs/validation-global-sdpa-failed.json','latency-experiment/logs/tail-default-allocator.json','latency-experiment/logs/tail-tuned-allocator.json','latency-experiment/logs/tail-blas-one.json','latency-experiment/cpu_attention_optimized.py'])
lat=read('latency-experiment/result-round2.json')
phase('R09','RMSNorm/卷积原生融合、chunk128、4→3线程','进一步降低CPU分配与线程竞争，处理周期慢峰。','自行实现ARM CPU RMSNorm/4抽头卷积融合，保留中间BF16舍入和回退；扫描分块筛选64/128/256，最终128。早期卷积被保守资格条件排除，不能把该次ACTIVE/PASSIVE对照当最终卷积对照，修复后显式检查norm_calls/conv_calls。早期PASSIVE轮在100次样本完成后因运行中脚本被覆盖而退出2，保留为探索证据；后续使用原子替换。临时HTTP也有过ready时序下连接中断；最终另行重跑验收。','最终4线程新进程100次中位823.92ms、P95838.21ms，最慢1396.80ms；3线程100次中位约865ms、P95约899ms、最慢约987ms，100/100低于1秒。另行HTTP100次中位814.30ms、P95826.02ms；计时人口不同，不混加。原健康检查Python进程与3个慢峰重叠，独立探针约0.9–1.0CPU秒，curl请求仅2–3ms；这是竞争证据，不是证明所有抖动只有该原因。','选择4个大核+Torch/OMP3，为系统活动留余量；164个主要算子用例及边界通过，7用例14问题最大差对原RKNN0.00363007、历史CPU0.00248951。没有靠关闭健康检查取成绩。两个已完成验证的临时HTTP会话在显式收尾时退出137，这是容器停止状态，不是已保存100次测试失败。',['latency-experiment/result-round2.json','latency-experiment/logs/cpu-tuning.json','latency-experiment/logs/conv-tuning.json','latency-experiment/logs/tail-native-default.json','latency-experiment/logs/tail-native-active.json','latency-experiment/logs/tail-native-passive.json','latency-experiment/logs/tail-final.json','latency-experiment/logs/tail-reserved.json','latency-experiment/logs/interference-analysis.json','latency-experiment/logs/health-client-cost.json','latency-experiment/logs/http-final-early-ready-failed.log'],table(['原始版本名','次数','中位数ms','P95 ms','最大ms'],[[e(k),v['count'],ms(v['median']),ms(v['p95_nearest_rank']),ms(v['maximum'])] for k,v in lat['experiments'].items()]))
olddeploy=read('optimized-deployment/deployment-result.json')
phase('R10','三线程版正式部署与Git c545f08','把通过验证的三线程候选变成可回退的服务。','先8901验收，再在8900启用decision3-edge-optimized；保留原RKNN和CPU容器。主线程队列、只读源码/模型、原生curl健康检查、restart=unless-stopped；实际回滚再恢复。','正式LAN HTTP100次中位859.21ms、P95870.34ms、最慢875.88ms，100/100低于1秒。此处是部署后HTTP，不能用上一轮814ms差异认定部署退化/优化因果。','成为后续独立实验始终保留的线上基线。2026-10-18才由最终graph-half替换；旧容器仍保留回退。',['optimized-deployment/deployment-result.json','optimized-deployment/server.py','optimized-deployment/run.sh','optimized-deployment/rollback.sh','optimized-deployment/cutover-verification.json','latency-experiment/logs/deployed-production-http.json'])
phase('R11','859ms细拆：NPU同步、桥接、扫描与分核微测','要到200ms，剩余CPU和桥接是否足够大，双核是否能重叠？','加入自有计时桥接，主线程exclusive分区闭合；同步区间包括提交/驱动等待，不冒充纯NPU计算。公开独立context双核实验按轮轮换：serial_split、parallel_same_core、parallel_split。','原模型无插桩前/后约868.99/851.01ms；插桩分区约SDK199.38ms、桥接/分配207.90ms、扫描254.52ms等。两矩阵微测串行5.621ms、同核并行4.277ms、分核并行3.888ms，约1.45倍只针对两矩阵。','确认自有CPU/桥接有优化空间，不能从微测声称整模型1.45倍。早期A53探针不稳定保留；把扫描和全部非Runtime桥接假想为0仍约437.8ms，只是当前路径记账，不是硬件下界。',['bottleneck-analysis/analysis-result.json','bottleneck-analysis/logs/profile-current.json','bottleneck-analysis/probe_parallel_interleaved.py'])
bridge=read('bridge-parallel-experiment/experiment-result.json')
phase('R12','桥接融合 / 同层独立投影并行 / 组合版','减少BF16↔FP16↔FP32转换、分配与调用，并让无依赖投影重叠。','融合版直接在自有RKNN缓冲做转换/有限值检查/输出BF16；并行版给MLP gate/up、MHA Q/K/V、Delta qkv/z独立context与持久工作线程，全部任务结束后才复用或抛错；组合版共享同输入准备，不做跨请求答案缓存。','30次顺序筛选baseline855.68、fusion699.21、parallel817.07、combined662.38ms；末尾baseline806.66ms，存在漂移，因此不能独立认领parallel收益。10轮正式成对baseline859.08→combined721.37ms，逐轮降时延中位16.03%；单引擎100次664.76ms、P95670.22ms。两引擎成对与单引擎条件不同。','组合版合格，后续以此优化扫描；7用例14问最大差对三线程0.00483379、历史CPU0.00489881。初版NaN规范值错误修复后重验；双引擎shadow撞1024 FD导致139，隔离容器提高至4096后复测通过，未改线上限制。',['bridge-parallel-experiment/experiment-result.json','bridge-parallel-experiment/logs/unit-attempt01-failed.log','bridge-parallel-experiment/logs/shadow-fd-limit-failed.log','bridge-parallel-experiment/logs/model-results.json','bridge-parallel-experiment/logs/paired-results.json','bridge-parallel-experiment/REPRODUCTION.txt'],table(['30次初筛版本','中位数ms','P95 ms','原passed'],[[e(k),ms(v['median']),ms(v['p95']),e(v['passed'])] for k,v in bridge['sequential_ablation'].items()]))
scan=read('scan-optimization-experiment/experiment-result.json')
phase('R13','CPU GatedDeltaNet：compact → native tile16/tile32','组合版中的扫描仍慢，能否在CPU重写同一递推？','紧凑布局与tile/fused原生循环，保持因果扫描、FP32状态和原权重。早两轮资格检查全部回退，不能当提速；随后发现FP32 decay非连续布局假设，修复真实输入重放后再跑。','12轮交替：baseline672.24ms、compact610.90、native_t16_f5569.13、native_t32_f5572.81。选择t16/f5，100次570.94ms、P95578.14ms；扫描轻量计时201.32→102.21ms，其余当前路径残差约469.16ms。','合格，7例14问最大差0.00312631，480项递推/边界检查。native支持1..512 token；初始/返回状态、packed、超长或不支持形状走原路径。469ms是扣除扫描的残差，绝不是整个模型已经469ms。',['scan-optimization-experiment/experiment-result.json','scan-optimization-experiment/logs/full-results.json','scan-optimization-experiment/logs/micro-ineligible-results.json','scan-optimization-experiment/logs/micro-ineligible-with-metadata.json','scan-optimization-experiment/logs/replay-layout-failed.log','scan-optimization-experiment/logs/replay-v1-results.json','scan-optimization-experiment/logs/replay-v2-results.json','scan-optimization-experiment/REPRODUCTION.txt'],table(['同引擎12轮版本','中位数ms','P95 ms'],[[e(k),ms(v['median']),ms(v['p95_nearest_rank'])] for k,v in scan['paired_summary'].items()]))
phase('R14','FP32输出→FP16输出；补上溢出CPU回退','约571ms中，SDK同步能否通过公开输出类型减少开销？','公开MatMul输出类型改为FP16，再回到BF16，输入/权重不量化。未保护版本发现1048576会变Inf，因此加入转换时非有限/饱和检测，该投影整段回退原CPU，验证并行组和后续恢复。CPU SiLU多项式/归一化与SLEEF路线另测。','未保护成对559.11→508.16ms，12/12更快；保护版严格成对564.96→512.93ms，6/6更快。最终safe-default100次中位512.25ms、P95519.07ms。CPU融合筛选baseline553.03、norm550.85、activation563.33、all576.61ms，无端到端收益。被动等待541.07ms较前后default509.93/509.54ms慢。','采用带范围保护的FP16输出，7例最大差0.00561126。不能从公开输入输出类型推断内部累加完全不变，也不称无损；CPU融合/SLEEF/被动等待未选。',['remainder-optimization-experiment/experiment-result.json','remainder-optimization-experiment/RESULTS.txt','remainder-optimization-experiment/logs/model-results.json','remainder-optimization-experiment/logs/sleef-screen.json','remainder-optimization-experiment/REPRODUCTION.txt'])
proj=read('projection-compile-experiment/experiment-result.json')
phase('R15','同输入投影合并、双核平衡、CPU大区域编译','在512ms安全版上减少多次提交/分列写回，或用CPU编译减少中间开销。','公开SDK合并同输入投影、等宽分核和原地scatter；CPU编译分别测动态MLP、大区域动态/固定签名、编译内核1/3线程。首次CPU编译丢失中间BF16舍入，修复为显式位运算后才继续；早期group探针出现KeyError，修正后重新保存group-probe结果。编译器runner分支也经修正后才补跑cpp1_suite，不把失败启动当候选成绩。','单引擎100次最佳509.67ms、P95516.74ms；20轮成对516.75→508.78ms，逐轮改善1.28%，18/20更快。CPU编译各版约524.70–620.92ms，未建立端到端收益，首次编译/初始化时间单列不计暖态。','只采用合并+等宽分核+直接scatter，78→72句柄；对安全512基线受检数值差0，对旧FP32输出参考最大0.00561126。编译路线全部不纳入最终候选，未改阈值迁就。',['projection-compile-experiment/experiment-result.json','projection-compile-experiment/RESULTS.txt','projection-compile-experiment/logs/scatter-balanced-none.json','projection-compile-experiment/logs/scatter-paired-balanced-none.json','projection-compile-experiment/REPRODUCTION.txt'])
phase('R16','约510→454ms：BTC卷积、扫描前处理、NEON转换','剩余转换和布局复制能否减少，扫描代数重排是否有效？','卷积直接读BTC避免转BCT复制；扫描前BF16→FP32、布局、Q/K norm和缩放合并一次遍历；转换常见值使用NEON整数位操作，特殊值仍旧路径。另试cacheable内存+手工同步、decay-late及readout-identity代数重排。','20轮成对507.88→454.56ms，10.09%、20/20更快；单引擎100次454.33ms、P95466.55ms。卷积局部重放55.47→27.31ms且逐元素一致。cache候选527.83ms比baseline511.77慢；两种扫描重排六层中位数和163.22/162.80ms，比原136.24ms慢。','采用卷积/前处理/转换组合；7例对510ms最大差0.00592196、旧FP32输出参考0.00793384。拒绝cache和代数重排；局部时间和整模型时间不混用。',['remaining-500ms-experiment/experiment-result.json','remaining-500ms-experiment/RESULTS.txt','remaining-500ms-experiment/logs/screen.json','remaining-500ms-experiment/logs/screen2.json','remaining-500ms-experiment/REPRODUCTION.txt'],screen('remaining-500ms-experiment/logs/screen.json')+screen('remaining-500ms-experiment/logs/screen2.json'))
phase('R17','完整注意力图与核心图：更快但数值门槛失败','用RKNN完整注意力子图是否能消除CPU注意力及边界？','完整层导出QKV、Q/K norm、RoPE、GQA、非因果mask、QK-softmax-V、gate、O投影；6层future-key/遮罩检查通过。严格BF16中间舍入ONNX因常量折叠bfloat16类型失败；FP16图可编译。另拆仅QK-softmax-V核心图，外围保留旧舍入。','16次初筛baseline453.96、scan435.99、完整graph398.14、graph+scan392.64ms，原7例通过；扩展12个同形状场景后，完整图最大差0.02795707，图+scan0.02297668，core0.02572922，core+scan0.01780373，均超过0.01。','全部注意力图拒绝上线。core v1把优化NHWC属性误当用户数组布局，先修为明确NCHW再作正式数值判断；旧大偏差作废。公开PERF_DETAIL辅助采集两次崩溃，未定位自有wrapper或SDK根因，不冒称已有逐算子报告。',['attention-chunk-experiment/RESULTS.txt','attention-chunk-experiment/experiment-result.json','attention-chunk-experiment/logs/core-io-diagnostic.json','attention-chunk-experiment/logs/core-v1-invalid-layout.json','attention-chunk-experiment/REPRODUCTION.txt'])
attn=read('attention-chunk-experiment/experiment-result.json')
phase('R18','CPU小块矩阵扫描四版：选BLAS2 block32/tile64','不用有精度问题的注意力图，能否把相同扫描递推小块矩阵化？','第一版/第二版自写块算法真实六层重放约70.03/72.10ms，分别慢于原63.04/65.41ms。接入NumPy捆绑OpenBLAS的SGEMM+STRSM，BLAS版57.94ms、BLAS2版52.12ms，对各自原65.14/61.67ms更快。外层OMP3，内层BLAS1，保持FP32状态、因果和回退。','20轮同模型72句柄：baseline452.65→scan442.88ms，逐轮2.15%，19/20更快；单候选100次442.32ms、P95454.43ms。原7例最大差对454版0.00334133、旧FP32输出0.00872152；额外12例对454版0.00511825。','只采用BLAS2块扫描，图全部排除。依赖固定镜像NumPy ILP64符号和路径；改变镜像/NumPy须重测。qualified-paired顶层默认标签曾写scan-core，实际逐次标签和selected均baseline/scan，已另行注明，未改原数值。',['attention-chunk-experiment/logs/chunk-unit-replay.json','attention-chunk-experiment/logs/chunk-v2-unit-replay.json','attention-chunk-experiment/logs/chunk-blas-unit-replay.json','attention-chunk-experiment/logs/chunk-blas2-unit-replay.json','attention-chunk-experiment/logs/qualified-paired.json','attention-chunk-experiment/logs/final-single-scan.json','attention-chunk-experiment/REPRODUCTION.txt'])
phase('R19','442ms后的LUT初版、布局修复、CPU注意力与小投影','新剖析发现SiLU、norm、完整注意力与桥接各有开销，逐个消融。','初版BF16全域查表：激活、MLP SiLU×up、norm分别试；发现激活输出布局变为BCT contiguous，与真实BTC-backed BCT排列不匹配，修复保持密集排列。CPU GQA注意力用FP32 SGEMM和稳定向量softmax，避免重复头复制；合并小a/b投影的同输入计算。','初版activation466.27ms且扩展差0.0101348失败，MLP410.94ms、norm427.81ms通过。修布局后activation400.54、MLP+norm400.21、向量attention428.80、小投影merge425.05、组合393.43ms。标量softmax432.02ms但最大差0.0170943超限。','保留有收益且通过扩展集的向量/LUT/小投影组合；拒绝标量softmax，布局失败原型不晋级。全65536种BF16激活、真实24次激活重放和边界验证；最初梯度测试错误嵌套inference_mode，修正测试环境后重跑，不将测试框架错误归咎模型。',['cpu-tail-experiment/logs/lut-screen.json','cpu-tail-experiment/logs/tail-screen.json','cpu-tail-experiment/logs/activation-replay.json','cpu-tail-experiment/logs/lut-unit-v1-test-harness.log','cpu-tail-experiment/PROGRESS.txt'],screen('cpu-tail-experiment/logs/lut-screen.json')+screen('cpu-tail-experiment/logs/tail-screen.json'))
phase('R20','CPU残差/norm/RoPE融合与MLP直接缓冲流水','继续削掉CPU和NPU之间的张量往返。','先融合BF16 add→norm、Q/K norm+部分RoPE、sigmoid gate，保留原舍入边界；再将自有gate/up原始FP16缓冲经原BF16舍入、SiLU LUT、乘法舍入后，直接写入down投影SDK缓冲。等宽gate/up不被旧pack合并时使用原独立句柄；任一错误等待全部worker后整MLP回退。','CPU组合筛选369.31ms；残差358.43、加RoPE组合356.10；完整attention prepare组合351.60ms；MLP pipeline324.58ms，对同轮CPU359.79ms更快。RoPE单独增量不稳定。所有是筛选，不能直接当最终正式速度。','采用融合与流水作为后续CPU候选和图回退。26域LUT检查、96注意力形状/mask、32 attention-prepare精确案例、12流水域检查及输出溢出/worker join/整段回退验证。',['cpu-tail-experiment/logs/fusion-screen.json','cpu-tail-experiment/logs/prepare-screen.json','cpu-tail-experiment/logs/pipeline-screen.json','cpu-tail-experiment/logs/pipeline-guards.json','cpu-tail-experiment/logs/pipeline-unit.json','cpu-tail-experiment/REPRODUCTION.txt'],screen('cpu-tail-experiment/logs/fusion-screen.json')+screen('cpu-tail-experiment/logs/prepare-screen.json')+screen('cpu-tail-experiment/logs/pipeline-screen.json'))
phase('R21','公开桥接布局、转换LUT、M115：负面或无因果收益','CPU已经更快，原生A/C布局或更短padding能否降低矩阵桥接？','对自有桥接尝试转换lookup、公开native A/C layout、M115专用尺寸。初版假设C按4-lane，查询公开属性发现FP16 C为[N/8,M,8]后修复；对照枚举矩阵/scatter。没有读取厂商私有结构或反编译。','同轮baseline329.66、lookup435.30、native438.11、M115356.13、末尾baseline361.60ms；各最终数值检查通过。','lookup/native明显更慢；M115处于前后baseline漂移范围，不能认领收益，均未采用。原布局假设失败归属自有wrapper，不归咎SDK。',['cpu-tail-experiment/logs/bridge-screen.json','cpu-tail-experiment/logs/bridge-modes-v1-assumed-C4.json','cpu-tail-experiment/logs/native-layout-attrs.json','cpu-tail-experiment/logs/bridge-modes-unit.json'],screen('cpu-tail-experiment/logs/bridge-screen.json'))
phase('R22','Mali-G52公开GPU路线：自有kernel → 私有状态扫描 → CLBlast','绕开闭源RKLLM，公开Panfrost/Rusticl能否加速注意力/扫描？','主机OpenCL无ICD时先只下载/解包Mesa/Rusticl/libclc等到私有目录，通过私有mount namespace运行，不替换系统图形驱动。自有OpenCL attention经历并行softmax Inf修复；分区扫描真实重放误差约0.105后改独立私有状态，再试volatile/unroll32与规定workgroup；最后用公开CLBlast批量SGEMM。GPU worker置CPU0–3，模型CPU4–7，IPC拷贝包含计时。','最终18 attention/6 synthetic/6 real-layer scan重放通过；GPU scan每层约305ms，六层约1.83s，阶段本身慢于整CPU模型，因此不再宣称有其整模型概率成绩。GPU attention整模型537.00ms，对同轮CPU331.88ms，数值通过（扩展最大0.00608767）。','公开GPU本次实现为负结果，未采用。原attention_mask关键字命名失配导致视觉请求异常，修复并补守卫再做整模型回归。没有由慢原型推断GPU理论极限；worker/socket/自己创建IPC清理完成。',['cpu-tail-experiment/logs/gpu-api-probe.json','cpu-tail-experiment/logs/gpu-private-runtime.json','cpu-tail-experiment/logs/gpu-kernels-v1-parallel-softmax.cl','cpu-tail-experiment/logs/gpu-unit-v2-partitioned.json','cpu-tail-experiment/logs/gpu-unit-v3-private-state.json','cpu-tail-experiment/logs/gpu-screen-v3-vision-keyword.log','cpu-tail-experiment/logs/gpu-unit.json','cpu-tail-experiment/logs/gpu-screen.json','cpu-tail-experiment/logs/cleanup.json'],screen('cpu-tail-experiment/logs/gpu-screen.json'))
phase('R23','MLP独立RKNN图：全12层FP32 I/O → 直接半精度I/O','完整注意力图精度不合格，不代表原权重MLP图也不能用。','捕获全部12层原MLP权重和真实115-token输入；复用既有x86 Docker、Torch2.4/RKNN2.5环境导出，不重建镜像。12个图均export_return=0。第一版F32 I/O再转换；HalfGraph直接BF16→FP16输入、原始FP16/FP32输出→BF16，减少冗余Torch缓冲。图初始化持久化、部分构造失败清理、越界/形状/权重变化回退。','第一版图352.44ms对同轮CPU346.35ms更慢，数值扩展最大0.00753716通过；直接I/O在12层真实输入与旧I/O逐位一致。五方案筛选CPU343.94、A72 CPU312.65、half图317.09、half+A72317.69ms。','保留half图等待正式复核，不能因为初筛A72更快就永久排除图。图只支持B1/T115，注意力继续原非因果CPU路径；原12层不剪、权重不训练、不量化，FP16执行引入的数值差单独验证。',['cpu-tail-experiment/logs/mlp-capture.json','cpu-tail-experiment/logs/compile-mlp-0-fp16graph.json','cpu-tail-experiment/logs/compile-mlp-11-fp16graph.json','cpu-tail-experiment/logs/mlp-graph-screen.json','cpu-tail-experiment/logs/mlp-half-io-unit.json','cpu-tail-experiment/logs/final-options-screen.json','cpu-tail-experiment/logs/mlp-graph-guards.json','cpu-tail-experiment/compile_mlp_suite.sh'],screen('cpu-tail-experiment/logs/mlp-graph-screen.json')+screen('cpu-tail-experiment/logs/final-options-screen.json'))
phase('R24','参数收敛：1/2/3/4线程、MKLDNN、spin、A72与图双核','更换融合实现后，参数初筛的改善是否稳定？','12轮轮换线程数，另测MKLDNN关闭；分别重启进程比较默认前/后、spin0/1000/10000、A72编译。公开图API的combined mask3实测支持（与早期MatMul接口不同），抽取0/5/11层测试。','线程3约336.47ms，1/2/4约344.29/343.37/343.99；关MKLDNN336.28差约0.05%，不认领。环境default前312.44、后324.89，spin0 348.04、1000 320.33、10000 347.76，未稳定胜过两侧default。图mask3聚合改善不足5%，不晋级整模型。','保留默认3线程、MKLDNN默认、默认spin。A72的早期9%收益未在正式对照中复现。一次scp覆盖正在运行的bash脚本造成命令偏移错误，旧数据归档，停止后原子上传并顺序重跑；这是编排错误，不是模型失败。',['cpu-tail-experiment/logs/thread-sweep.json','cpu-tail-experiment/logs/environment-default-before.json','cpu-tail-experiment/logs/environment-default-after.json','cpu-tail-experiment/logs/environment-spin0.json','cpu-tail-experiment/logs/environment-spin1000.json','cpu-tail-experiment/logs/environment-spin10000.json','cpu-tail-experiment/logs/mlp-core-probe.json','cpu-tail-experiment/logs/environment-suite-v1-live-script-update.log','cpu-tail-experiment/select_runtime.py'])
phase('R25','正式CPU版 → 补做图成对 → 最终318.51ms','必须用交替和100次复核筛选结果，排除顺序/进程漂移。','CPU正式20轮轮换442基线/CPU std/CPU A72；A72差不足1%优先默认。发现初筛排除MLP图的依据受A72假收益影响，补做CPU/half图/half+A72 20轮，并对选中图新进程100次、原7例14问+12额外场景+36复测、10次闭合分段剖析。','CPU20轮440.08→333.84ms，逐轮24.16%、20/20更快；CPU100次340.40ms、P95344.44。补充20轮CPU337.93→half图318.55ms，约5.98%、20/20更快；half+A72319.82ms。最终图100次318.51ms、P95328.75，最小294.09、最大346.75ms。','最终选graph-half。原7例对442版概率/score/noul最大0.00388647、旧FP32输出参考0.00483505；额外12例0.00753716，choice一致。方法调用不含HTTP。剖析均值330.36ms，其中MLP图76.31、scan61.48、完整attention38.52、其他SDK同步34.02、桥接余项28.45ms；未达200ms，不把当前残差当理论下界。',['cpu-tail-experiment/logs/formal-paired.json','cpu-tail-experiment/logs/formal-single.json','cpu-tail-experiment/logs/final-graph-comparison.json','cpu-tail-experiment/logs/formal-graph-single.json','cpu-tail-experiment/logs/final-graph-profile.json','cpu-tail-experiment/RESULTS.txt','cpu-tail-experiment/report.html'])
phase('R26','最终graph-half部署：318ms方法版 → 325ms真实HTTP','将候选真正替换三线程服务，并给出可回退、可核验的运行状态。','用户明确授权后创建不可变release，核对39项运行代码/库/图、冻结550文件运行依赖，模型只读；8901先验收，再停止旧三线程、8900启动新decision3-edge-graph，主线程串行执行与12图常驻。实际演练新→旧→新。','临时HTTP20次中位322.73ms；正式LAN HTTP100次中位325.27ms、P95328.92、最小321.50、最大340.45。19场景26问题含confidence与冻结候选完全一致（最大差0），115-token每次12图命中，其他长度/图片走CPU回退；422、模型列表、两个不同并发请求隔离通过。','发布2026-10-18-graph-half-v1；旧三线程容器停止保留restart=no，新服务healthy/unless-stopped。所有源/复现/报告已于682dd74推到main，模型/SDK/ARM库/图/编译venv不提交。当前记录是保存的部署验收快照，不是实时状态监控。',['cpu-tail-deployment/deployment-result.json','cpu-tail-deployment/production-http.json','cpu-tail-deployment/staging-http.json','cpu-tail-deployment/rollback-rehearsal.json','cpu-tail-deployment/package-manifest.json','cpu-tail-deployment/README.md','cpu-tail-deployment/VERIFY.txt'])
phase('R27','最终勘误：confidence并未被旧0.01比较器覆盖','历史“输出差异≤0.01”的说法到底比较了哪些字段？','发布前独立遍历原输出重新审计，确认旧compare跳过confidence及legend。保留所有历史数值，不倒改参考，不声称通过一个未做过的检查；新HTTP与冻结候选等价性检查包含confidence。','概率/score/noul最大0.00753716；confidence相对442候选最大0.01002684019，略超过0.01，应独立披露。新HTTP相对冻结graph-half全数值最大差0，两种比较参考不同，不能混为同一个结论。','修正HTML、复现说明和部署文档；choice一致不等于概率完全相同，输出一致性也不等于真实标注任务准确率。没有以放宽概率门槛来接纳失败注意力图。',['cpu-tail-deployment/confidence-audit.json','cpu-tail-deployment/audit_confidence.py','remainder-optimization-experiment/common.py','cpu-tail-deployment/verify_http.py'])

# Every retained evidence/configuration/source snapshot is catalogued. Individual
# timing repetitions are not separate versions; their original files stay linked.
tracked=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=ROOT,text=True).splitlines()
def group_of(n):return n.split('/')[0] if '/' in n else 'root'
def in_scope(n):
 if n in SELF or group_of(n) not in GROUP_NAMES:return False
 return Path(n).suffix in ('.json','.log','.txt','.md','.py','.sh','.cpp','.c','.h','.hpp','.cl','.html') or Path(n).name in ('Dockerfile','Dockerfile.server','Dockerfile.cpu','requirements.lock')
files=sorted({n for n in tracked if in_scope(n) and (ROOT/n).is_file()})
json_files=[n for n in files if n.endswith('.json')]
parsed={n:read(n) for n in json_files}
SKIP={'sources','source_sha256','source_hashes','hashes','source_sha256_output','verified_runtime_hashes','verified_runtime_source_hashes','tested_directory_snapshot','package_manifest','baseline_cases','extra_refs','response','responses','answers','probabilities','legend','criteria','state','images','videos','extra_cases','all_cases','records','requests','samples','seconds','raw','profiles','native_handles','worker_intervals','case_responses','first_signature_seconds','aten','events','config','configs','settings','captured_metadata','metadata','weight_keys_preview','loaded_config','cases','extra','extra_repeats','equation_checks','checks','assertions','unit_results','error_records','checkpoint','mounts','health','final_health','initial_health','Config','HostConfig','input','inputs'}
METRIC=re.compile(r'(?:^|_)(?:median|p95|min|max|count|delta|speedup|return_code|return|cases|questions|unit_cases|faster_rounds|seconds|ms|bytes|failure|matched|total|difference)(?:_|$)')
STATUS={'passed','completed','selected','selected_variant','accepted','status','stage','variant','backend','tag','export_return','build_return_code','load_return_code','parity_passed','worth_model_test','combined_supported','runtime_initialization','conversion','native_runtime_fixed','new_model_speedup_measured','new_native_model_inference_performed','target_reached','selected_environment'}
entries=[]
def walk(v,n,path='',ancestral_kind=''):
 if isinstance(v,dict):
  fields={}
  for k,x in v.items():
   if k in SKIP:continue
   if isinstance(x,(str,int,float,bool)) and (k in STATUS or METRIC.search(k)):
    # These are literal source fields, not inferred acceptance decisions.
    fields[k]=x
  for bucket in ('timing','summary'):
   child=v.get(bucket)
   if isinstance(child,dict):
    for k,x in child.items():
     if isinstance(x,(int,float,bool,str)) and (METRIC.search(k) or k in STATUS):fields[bucket+'.'+k]=x
  # Collapse per-question and per-unit repetitions into version-level extrema.
  for bucket in ('cases','extra','extra_repeats'):
   if bucket not in v:continue
   maxima={};checks=[]
   def quality(x,trail=()):
    if isinstance(x,dict):
     for k,y in x.items():
      if k in ('response','answers','probabilities','legend'):continue
      if k=='passed' and isinstance(y,bool):checks.append(y)
      if k.startswith('max_') and isinstance(y,(int,float)):
       ref=next((part for part in trail if part.startswith('vs_')), 'reported')
       key=ref+'.'+k;maxima[key]=max(maxima.get(key,-float('inf')),y)
      elif isinstance(y,(dict,list)):quality(y,trail+(k,))
    elif isinstance(x,list):
     for y in x:quality(y,trail)
   quality(v[bucket])
   for k,y in maxima.items():fields[bucket+'.'+k]=y
   if checks:fields[bucket+'.all_reported_passed']=all(checks)
  if not fields and re.search(r'/variants/[^/]+$',path):fields={'configuration_record':'无统一汇总字段；见原始记录，不自动推断结果'}
  if fields:
   entries.append({'file':n,'pointer':path or '/','fields':fields})
  for k,x in v.items():
   if k in SKIP:continue
   if k in ('timing','summary') and isinstance(x,dict) and any(bucket+'.' in z for bucket in (k,) for z in fields):continue
   if k=='sum_layer_medians_ms' and isinstance(x,dict):
    for name,val in x.items():
     if isinstance(val,(int,float)):entries.append({'file':n,'pointer':path+'/'+k+'/'+name,'fields':{'sum_layer_medians_ms':val,'scope':'局部各层中位数之和；不是整模型'}})
    continue
   if isinstance(x,(dict,list)):walk(x,n,path+'/'+k.replace('~','~0').replace('/','~1'))
 elif isinstance(v,list):
  for i,x in enumerate(v):
   if isinstance(x,(dict,list)):walk(x,n,path+'/'+str(i))
for n,d in parsed.items():walk(d,n)

# Main summary tables cite their own source, never a best-of-mixed-protocol gain.
progress_rows=[
 ['官方BF16重启抽查','HTTP单次；115token','15553.00','—',link('restart-results.json')],
 ['FP32 CPU适配','HTTP单次；115token','7170.00','—',link('optimized-production-result.json')],
 ['初版RKNN混合','HTTP3次中位；115token','2822.27','未提供',link('rknn-matmul-experiment/experiment-result.json')],
 ['三线程正式部署','HTTP100次；115token','859.21','870.34',link('optimized-deployment/deployment-result.json')],
 ['桥接融合+投影并行','模型内100次；115token','664.76','670.22',link('bridge-parallel-experiment/experiment-result.json')],
 ['原生Delta扫描t16/f5','模型内100次；115token','570.94','578.14',link('scan-optimization-experiment/experiment-result.json')],
 ['安全FP16输出','模型内100次；115token','512.25','519.07',link('remainder-optimization-experiment/experiment-result.json')],
 ['投影合并+平衡+scatter','模型内100次；115token','509.67','516.74',link('projection-compile-experiment/experiment-result.json')],
 ['BTC卷积+前处理+转换','模型内100次；115token','454.33','466.55',link('remaining-500ms-experiment/experiment-result.json')],
 ['BLAS2块扫描','模型内100次；115token','442.32','454.43',link('attention-chunk-experiment/experiment-result.json')],
 ['最终CPU融合版','模型内100次；115token','340.40','344.44',link('cpu-tail-experiment/logs/formal-single.json')],
 ['最终graph-half候选','模型内100次；115token','318.51','328.75',link('cpu-tail-experiment/logs/formal-graph-single.json')],
 ['最终graph-half上线','LAN HTTP100次；115token',ms(latest['http']['timing']['median']),ms(latest['http']['timing']['p95_nearest_rank']),link('cpu-tail-deployment/production-http.json')],
]
summary_table=table(['阶段版本','计时口径','中位/单次 ms','P95 ms','证据'],[[e(c) if i<4 else c for i,c in enumerate(row)] for row in progress_rows])

# Compact all structured outcomes without rewriting historical metric names/units.
def fields_html(d):
 return '<br>'.join('<code>'+e(k)+'</code> = '+e(scalar(v)) for k,v in d.items())
appendix=[]
for group,title in GROUPS:
 items=[x for x in entries if group_of(x['file'])==group]
 table_rows=[[link(x['file']),'<code>'+e(x['pointer'])+'</code>',fields_html(x['fields'])] for x in items]
 appendix.append(f'<details class="appendix"><summary>{e(title)}：{len(items)}条结构化结果/验证状态</summary>'+table(['原始JSON','JSON路径（原配置/版本）','原始字段与数值'],table_rows)+'</details>')

# Raw failure logs and source snapshots are first-class records, not silently dropped.
fail_files=[n for n in files if re.search(r'(?:failed|invalid|ineligible|attempt\d|aborted|crash|live-script-update|v[123](?:[-.]|$))',Path(n).name,re.I)]
failure_table=table(['记录','类型','解释边界'],[[link(n),e(Path(n).suffix or '文件'),'名字提示失败/旧版；实际原因与是否重跑以R01–R27及原文为准，不能只凭文件名判定SDK故障。'] for n in fail_files])
inventory=[]
for group,title in GROUPS:
 group_files=[n for n in files if group_of(n)==group]
 rows=[]
 for n in group_files:
  p=ROOT/n
  if '/logs/' in n and p.suffix in ('.py','.c','.cpp','.cl','.h','.sh','.hpp'):kind='留存源码版本/快照'
  elif p.suffix=='.log':kind='原始运行日志（含失败/中断）'
  elif p.suffix=='.json':kind='结构化结果、配置或数值参考'
  elif p.suffix in ('.txt','.md','.html'):kind='阶段说明、复现/报告或过程记录'
  else:kind='实现、构建、测试或服务脚本'
  digest=hashlib.sha256(p.read_bytes()).hexdigest()
  rows.append([link(n),e(kind),p.stat().st_size,'<code>'+digest+'</code>'])
 inventory.append(f'<details class="appendix"><summary>{e(title)}：{len(group_files)}个证据/源码文件</summary>'+table(['文件','归类','字节','SHA256（本记录生成时）'],rows)+'</details>')

nav=''.join(f'<a href="#{code}">{code} {e(title)}</a>' for code,title,_ in sections)
commit_rows=[['ecefc20','基础部署研究及独立安装文档'],['c545f08','三线程优化正式发布与回退'],['682dd74','最终graph-half服务、前置优化源码、证据及HTML报告']]
commit_table=table(['历史提交','主要内容'],[[f'<a href="https://github.com/griffith-h/decision3-edge-deployment/commit/{commit}"><code>{commit}</code></a>',e(desc)] for commit,desc in commit_rows])
style='''body{margin:0;background:#f4f6fa;color:#15283e;font:16px/1.8 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1140px;margin:auto;padding:36px 24px 70px}h1{line-height:1.3;font-size:34px}h2{line-height:1.4;font-size:25px}h3{font-size:20px}section{margin-top:28px;padding:26px;background:#fff;border:1px solid #dce3ec;border-radius:12px;scroll-margin-top:20px}.lead,.notice{background:#e9f1fc;border-left:4px solid #2f6594;padding:16px 20px}.eyebrow,.small{font-size:13px;color:#566a80}.sources{font-size:13px;overflow-wrap:anywhere;border-top:1px solid #dce3ec;padding-top:14px}a{color:#095e9c}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere;font-size:.85em}table{width:100%;border-collapse:collapse;font-size:13px;line-height:1.65;background:white}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #dce3ec;padding:10px}th{background:#eaf0f7}td:first-child{min-width:140px;overflow-wrap:anywhere}td{overflow-wrap:anywhere}.scroll{overflow:auto}.appendix{margin:14px 0;padding:14px;background:white;border:1px solid #dce3ec;border-radius:8px}summary{cursor:pointer;font-weight:650}.toc{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:6px 18px;font-size:14px}.toc a{display:block}.search{position:sticky;top:0;z-index:2;background:#f4f6faf5;padding:12px 0}.search input{box-sizing:border-box;width:100%;padding:12px;border:1px solid #a7bbce;border-radius:7px;font-size:16px}.hidden{display:none!important}.count{font-size:13px;color:#526981}.pill{display:inline-block;border-radius:6px;background:#dbe9fa;padding:3px 8px;font-size:13px}.muted{color:#526981}footer{padding-top:30px;font-size:13px;color:#526981}@media(max-width:600px){main{padding:20px 12px 40px}section{padding:17px}h1{font-size:28px}h2{font-size:22px}td,th{padding:8px}.toc{grid-template-columns:1fr}}@media print{.search,button{display:none}body{background:white}main{max-width:none;padding:0}section{break-inside:avoid}details{break-inside:auto}a{color:inherit}}'''
body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>d3-edge × RK3576 全过程版本与尝试记录</title><style>{style}</style></head><body><main>
<p class="pill">2026-10-13 至 2026-10-18 · 记录截至提交682dd74后的最终部署验收</p><h1>d3-edge × RK3576<br>从初次CPU部署到graph-half的全部留存尝试</h1>
<p class="lead">这是一份统一记录：<b>{len(sections)}个主阶段、{len(entries)}条结构化结果/验证状态、{len(files)}个留存证据/源码文件</b>。主文说明每一步为什么做、怎么改、测到什么、为什么采用或放弃；附录逐项列出保存的配置、失败日志与版本快照。主文、阶段表、附录都在这一个HTML中，无外部脚本/字体依赖；相对链接用于在完整仓库中打开原始证据。</p>
<p>这里的R01–R27是本记录的阅读编号，<b>不是新Git tag或27个正式发布版本</b>。试验分支会交错运行，按实际依赖与推进顺序叙述；缺少精确时间戳的原型不虚构分钟级时间线。日期按Asia/Shanghai阅读；JSON里的UTC保留原样。</p>
<div class="notice"><b>最终结果：</b>完整12层、原权重与注意力语义保留；graph-half暖态方法100次中位318.51ms，正式LAN HTTP100次中位{ms(latest['http']['timing']['median'])}ms、P95{ms(latest['http']['timing']['p95_nearest_rank'])}ms。图快路径只覆盖batch1、115 token，其他形状CPU回退。旧三线程服务停止保留且回滚演练通过，200ms未达成。本状态来自{e(latest['verified_at_utc'])}的保存验收，不是实时监控。</div>
<div class="search"><label for="find">筛选表格配置 / 文件名（正文可用浏览器查找）</label><input id="find" type="search" placeholder="例如：W8A8、512、compile、confidence、GPU、native_t16_f5"><span id="search-status" class="count">原始逐次样本保存在JSON中；附录不把每次计时冒充一个版本。</span></div>
<h2 id="rules">先统一比较口径</h2>
<ol><li><b>不改模型：</b>采用的路径均保留官方发布的12层与权重，完整注意力非因果、DeltaNet扫描因果。上游发布模型本来就有不规则层排列，并不是我们把它再裁成12层。唯一18槽位RKLLM是未采用的恒等padding兼容诊断。</li><li><b>计时分四类：</b>整模型方法、LAN HTTP、局部/合成微基准、转换/编译/初始化。各自标注请求长度、问题数、暖机和资源条件；不把它们串成统一倍数，也不挑掉最慢样本。P95按nearest-rank。</li><li><b>精度不是准确率：</b>历史大多数compare只检查概率/score/noul并排除confidence；原0.01门槛按实际实现表述。最终独立审计confidence最大0.01002684，另列；新HTTP对冻结候选的全字段数值差0。所有“通过”仍只覆盖已测样本。</li><li><b>阶段胜出不等于上线：</b>仅初始CPU、CPU适配、原RKNN、三线程和最终graph-half曾提供服务；后续大部分版本在独立进程测试。阶段RESULTS中的“未部署/未推送/线上保持”是当时状态，后来统一提交或上线并不倒改历史日志。</li><li><b>结论边界：</b>没有厂家二进制反编译，没有新增剪层/训练；公开API与自行实现的CPU/桥接/GPU代码可改，不代表能修闭源执行器。失败先区分自有代码、测试框架、编排和未定位SDK边界，不把自有bug算成厂家硬件极限。</li></ol>
<h2 id="milestones">主要可比测量里程碑</h2><p>本表提供定位，<b>不同协议的行不用于计算端到端累计提速</b>。每阶段的成对证据在主文，所有其他候选在附录。</p>{summary_table}
<h2 id="toc">完整阶段目录</h2><nav class="toc">{nav}</nav>
{''.join(content for _,_,content in sections)}
<section id="final-decision"><h2>现在留下了什么、没有证明什么</h2><p>最终采用的是CPU融合、公开MatMul持久句柄、完整非因果CPU注意力、BLAS2因果扫描，以及只针对B1/T115的12个MLP图。并没有使用那个更快但概率超限的完整注意力图，也没有使用语义错误的RKLLM原型。</p><p>200ms在本次硬件、固定SDK与已验证实现中没有达成；分散的NPU调用/等待、MLP图、扫描、完整注意力、桥接及其他CPU工作仍占时间。不能把某轮“删掉扫描后的剩余时间”说成理论下界，也不能证明未来任何原生执行器/编译器/硬件都无优化空间。</p><p>没有新的带标签任务准确率评测、视频HTTP回归、冷启动SLA或并发吞吐承诺。新机器基础安装与318ms升级仍需准备固定镜像、SDK、模型、历史依赖和另行生成的ARM库/图；仅克隆代码不是已安装成功。</p><p>完整实际部署与回退步骤见{link('cpu-tail-deployment/README.md','最终部署说明')}；逐次服务验收见{link('cpu-tail-deployment/production-http.json')}；后期可读图表见{link('cpu-tail-experiment/report.html','最终HTML报告')}；基础安装看{link('d3-edge-rk3576-install-guide.html','独立安装指南')}。</p>{commit_table}</section>
<section id="config-records"><h2>附录A：全部留存JSON的版本级结果摘要</h2><p><b>共{len(json_files)}个JSON文件，提取{len(entries)}条记录。</b>保留原始字段名与单位：median/seconds通常是秒，带_ms的字段是毫秒，sum_layer_medians_ms是局部各层中位数之和；原表若未提供单位/统计量，不自行补造。每行JSON路径给出原配置定位。</p><p>这里的passed/completed/selected等是源文件原字段，不自动等于“可上线”；例如首次7用例通过的注意力图后来在扩展12例失败，必须按R17的完整资格检查决定。单次records/requests样本、每个问题和方程单测不逐条伪装为版本；cases/extra聚合为该版本的最大差异及原passed字段合取，单位/参考名保留。完整逐次输出与单测仍在原文件中。</p>{''.join(appendix)}</section>
<section id="failed-records"><h2>附录B：失败、中断、无效布局与旧版快照索引</h2><p>名字包含failed/invalid/ineligible/attempt/v1/v2/v3等的{len(fail_files)}个文件在此集中列出。它是辅助检索，不是所有失败的唯一判据；例如数值不合格但文件名正常的完整注意力图在主文及附录A中保留。</p>{failure_table}</section>
<section id="inventory"><h2>附录C：全过程证据与源代码版本目录</h2><p><b>{len(files)}个文件逐一列出生成时SHA256。</b>包括已提交源码、原始日志、JSON参考、阶段说明与logs中的旧版源快照。只索引本项目根目录与上述16组研究/部署目录，不读取被忽略的模型权重、SDK二进制、编译venv或临时GPU依赖。</p><p>“全部”限定为本仓库确有留存证据的尝试；口头讨论而未实现/运行的方向标为未验证，不补写不存在的测量。缺少结构化JSON的日志和代码版本不会因没有分数而消失。</p>{''.join(inventory)}</section>
<footer>生成器：build_development_history.py。仅整理已有资料，不启动模型或更改服务。复核：python3 build_development_history.py --check。结构化JSON均解析、相对链接与锚点校验、原始证据SHA留存；本次文档工作不重新跑NPU性能试验。</footer></main>
<script>
const input=document.getElementById('find');const status=document.getElementById('search-status');
input.addEventListener('input',()=>{{const q=input.value.trim().toLocaleLowerCase();let hits=0;document.querySelectorAll('tr').forEach(row=>{{if(row.parentElement.tagName==='THEAD')return;const match=!q||row.textContent.toLocaleLowerCase().includes(q);row.classList.toggle('hidden',!match);if(match&&q)hits++;}});document.querySelectorAll('details.appendix').forEach(d=>{{const rows=[...d.querySelectorAll('tbody tr')];const match=!q||rows.some(r=>!r.classList.contains('hidden'));d.classList.toggle('hidden',!match);if(q&&match)d.open=true;}});status.textContent=q?`表格匹配 ${{hits}} 行；正文保留完整，可使用浏览器查找。`:'筛选已清空；原始逐次样本保存在JSON中。';}});
</script></body></html>'''

# Documentation validation only: links/anchors, deterministic generation, and
# representative milestones keyed to original numeric JSON, not live service.
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
class Audit(HTMLParser):
 def __init__(self):super().__init__();self.links=[];self.ids=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k=='id':self.ids.append(v)
   if k=='href':self.links.append(v)
audit=Audit();audit.feed(body)
assert len(audit.ids)==len(set(audit.ids))
for href in audit.links:
 u=urlsplit(href)
 if not u.scheme and u.path:assert (ROOT/unquote(u.path)).is_file(),href
 if not u.path and u.fragment:assert unquote(u.fragment) in audit.ids,href
assert len(sections)==27 and len({x['file'] for x in entries})>150
assert abs(read('cpu-tail-experiment/logs/formal-graph-single.json')['timing']['median']-.3185088306199759)<1e-12
assert latest['passed'] and latest['http']['timing']['count']==100
assert read('cpu-tail-deployment/confidence-audit.json')['max_confidence_delta_vs_442']>.01
for k in ['0.752699','0.02795707','469ms','318.51','325.27','confidence','没有进行反编译','512','编排错误']:
 assert k in body,k
args=argparse.ArgumentParser();args.add_argument('--check',action='store_true');a=args.parse_args()
if a.check:assert OUT.read_text()==body,'History is stale; regenerate it'
else:OUT.write_text(body)
verification={'passed':True,'document':OUT.name,'main_stages':len(sections),'evidence_files':len(files),'json_files_parsed':len(json_files),'structured_records':len(entries),'failure_and_legacy_files':len(fail_files),'relative_links_checked':sum(not urlsplit(h).scheme for h in audit.links),'anchors_unique':True,'historical_scope':'retained repository evidence through graph-half deployment at commit682dd74','service_or_model_execution':False,'document_sha256':hashlib.sha256(body.encode()).hexdigest()}
if not a.check:(ROOT/'history-verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(verification,ensure_ascii=False))
