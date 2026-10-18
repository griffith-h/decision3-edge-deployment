"""Preserve five historical stages; prepend measured sixth-stage evidence."""
from pathlib import Path
import html,json,statistics,subprocess,sys
from validate_records import stats
P=Path(__file__).resolve().parent;root=P.parent
load=lambda p:json.loads((P/p).read_text());esc=lambda v:html.escape(str(v));ms=lambda x:f'{x*1000:.2f}'
s=load('logs/summary.json');assert s['passed'];f=[load(f'logs/formal-{i}.json') for i in (1,2,3)];profile=load('logs/profile-native.json');reg=load('logs/regression-native.json');short=load('logs/short-formal.json');service=load('service-after.json');b,c=s['baseline'],s['candidate']
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def link(name):return f'<a href="../bf16-pipeline-experiment/{name}">{esc(name)}</a>'
formal=[]
for r in f:
 bb,cc=r['timing']['baseline'],r['timing']['candidate'];formal.append([r['round'],r['pairs'],ms(bb['median']),ms(bb['p95_nearest_rank']),ms(cc['median']),ms(cc['p95_nearest_rank']),esc(str(r['handles_before']))+' → '+esc(str(r['handles_after']))+' → '+esc(str(r['handles_after_close'])),str(r['fds_before'])+' → '+str(r['fds_after'])+' → '+str(r['fds_after_close']),link(f'logs/formal-{r["round"]}.json')])
formal.append(['合并',s['pairs'],ms(b['median']),ms(b['p95_nearest_rank']),ms(c['median']),ms(c['p95_nearest_rank']),'engine/native MLP/attention','暖态FD不变',''])
accuracy=[[name,f"{s['regression_maxima'][key]:.8f}",gate] for name,key,gate in [('分类概率','probability_max_delta','≤0.01'),('score（范围0–2）','score_max_delta','≤0.01'),('noul','noul_max_delta','≤0.01'),('confidence','confidence_max_delta','独立报告，不纳入上述门槛')]]
components=[]
for label,tag in [('完整注意力入口','attention_entry'),('扫描入口（含准备）','scan_entry'),('原生扫描（已含在入口内）','scan_c'),('其他大型投影','projection_rows'),('MLP入口','mlp'),('native MLP入口（已含在MLP内）','native_mlp')]:
 components.append([label,*[ms(statistics.median(x['components'].get(tag,{}).get('inclusive',0) for x in profile['records'] if x['mode']==m)) for m in ('baseline','candidate')]])
negative=[]
for filename,label in [('screen-g128-t3','扫描128行消费流水'),('screen-g256-t3','扫描256行消费流水'),('screen-g512-t3','扫描512行消费流水'),('screen-cross-g256-t3','三处下一全注意力层Q/K/V预取'),('screen-native-projections','native MLP及其他投影扩展'),('screen-native-stream-g256-t3','native MLP与256行扫描流水'),('screen-split','native MLP及320/295行down'),('screen-split-stream-g256-t3','并行down与扫描流水组合')]:
 r=load(f'logs/{filename}.json');negative.append([esc(label),' / '.join(esc(m)+': '+ms(v['median']) for m,v in r['timing'].items()),link(f'logs/{filename}.json')])
without_attention=statistics.median(x['seconds']-x['components'].get('attention_entry',{}).get('inclusive',0) for x in profile['records'] if x['mode']=='candidate')
section=f'''<section id="bf16-pipeline" class="box"><h2>第六阶段：保留BF16边界的MLP融合与跨层流水</h2>
<p><b>{'目标达标' if s['goal_met'] else '未达到1秒目标'}。</b>三轮独立容器、每轮30组暖态交错对照，候选90次P50 <b>{ms(c['median'])}ms</b>、P95 <b>{ms(c['p95_nearest_rank'])}ms</b>；上一合格候选对照为{ms(b['median'])}ms / {ms(b['p95_nearest_rank'])}ms，P50下降{s['reduction']*100:.2f}%。{s['paired_candidate_wins']}/{s['pairs']}组更快。615 token、batch1模型方法调用，不是HTTP/并发SLA。全部慢样本保留；未部署；研究成果随本次main提交归档。</p>
<h3>采用机制及范围</h3>
<p>使用原权重的公开FP16×FP16→FP32 MatMul。新自有桥接直接读取native FP32 C布局，依次保留gate/up的BF16舍入、原BF16 SiLU查表和BF16乘法舍入，再直接写入down的native A；没有先解包整个gate/up矩阵。down通过两个独立句柄、公开mask1/2分别处理320与295行，按原行序合并全部615行。所有中间输出保持私有，任一worker失败先等待全部worker，再整层原CPU MLP重算；故障层持续CPU回退。没有使用未量化BF16输出枚举或更改权重。长度域切换时，在全部worker完成之后退役闲置的半精度缓存或旧FP32 MLP对照缓存，保留新MLP暖态句柄；纯615计时不触发退役。</p>
<p>实际公开SDK查询确认native A为(K/8,M,8)、FP32 C为(N/4,M,4)，每个新上下文都会核对返回布局/大小。历史115行native FP16 C整体更慢，本轮的不同点是FP32类型和C→A之间的BF16融合，不能把它归因为单纯扩大shape。权重、12层、全部K/V、非因果mask及原32行block/64列tile扫描和跨块状态保持；仍只在受保护B1/T615 logits内选择该路径，其他形状沿用上一合格路径。</p>
<p>{link('selection.json')} · {link('cache_owner.py')} · {link('logs/cache-integration.json')} · {link('native_split.cpp')} · {link('split_mlp.py')} · {link('logs/native-fp32-attrs.json')} · {link('logs/split-mlp-unit.json')} · {link('logs/split-guards.json')}</p>
<h3>正式交错对照</h3>{table(['轮次','配对组数','对照P50 ms','对照P95 ms','候选P50 ms','候选P95 ms','句柄：暖机/结束/关闭','FD：暖机/结束/关闭','原始记录'],formal)}
<p>每模式先三次暖机，配对顺序逐组交替；同容器两模式持有相同资源银行，三轮为独立容器。P95采用nearest-rank；正式计时没有插桩，哈希、HTTP检查、编译与实验顺序分离。最大进程RSS约{s['formal_rss_max_kib']/1024/1024:.2f}GiB，不代表所有驱动/NPU内存。暖态FD和句柄稳定，关闭后各类自有MatMul句柄均为0。独立三轮bank/worker池创建释放后FD数为{esc(s['bank_lifecycle_fd_counts'])}，没有增长；设备、CPU识别/perf map等进程运行时FD可以保留至进程退出，未声称FD总数归零。未验证多Owner/并发。</p>
<p>115 token另做12组控制：对照P50 {ms(short['timing']['baseline']['median'])}ms，候选{ms(short['timing']['candidate']['median'])}ms；新MLP路径调用数均为0。{link('logs/short-formal.json')}</p>
<h3>精度与生命周期</h3>
<p>34例全部重新推理，并与第五阶段独立生成、冻结SHA的原current参照核对；本轮另新算615 current并断言与冻结参照完全相同。choice全部一致，原数值门槛全部通过。相对上一合格候选全部受检数值的最大差为{s['max_delta_vs_previous_qualified']:.8f}。score差不是概率差；confidence独立报告；这些结果不等于业务准确率。</p>
{table(['字段','相对原current最大绝对差','验收口径'],accuracy)}
<p>真实层回放0/5/11要求逐位一致；范围保护包括原FP16安全区间、非有限数、输入/乘积溢出。故障测试覆盖gate/up及down的全部等待、第二块失败后完整CPU重算、故障层持续CPU、部分创建清理、重复关闭，以及扫描生产者/消费者失败后整层重算和线程join。原始记录：{link('logs/bank-lifecycle.json')} · {link('logs/regression-native.json')} · {link('logs/native-guards.json')} · {link('logs/split-guards.json')} · {link('logs/stream-guards.json')}。</p>
<h3>独立profile与条件性结论</h3>
<p>每模式5次独立插桩，不混入正式计时。入口inclusive含子项，不能重复相加；分项中位数不要求相加等于总中位数。worker区间重叠，SDK区间包括提交、驱动、等待及转换，并非纯NPU计算。</p>
{table(['组件','上一合格候选ms','本轮候选ms'],components)}
<p>剩余成本仍分布在大型投影、完整扫描、MLP及CPU准备/残差。仅假设注意力耗时为零，同一批profile的其余时间中位数仍约{ms(without_attention)}ms；这只是当前实现的记账，不是硬件下限。进入1秒还须从本轮P50削减约{max(0,c['median']-1)*1000:.0f}ms，并同时压低P95。已测的分段/跨层预取没有稳定胜过本轮最终路径；新的全段调度或编译机制需要重新证明BF16边界、全K/V和生命周期，当前数据不能承诺其收益。</p>
<p>{link('logs/profile-native.json')} · {link('logs/summary.json')}</p>
<h3>未采用路线及失败记录</h3>
<p>以下均为六组/模式的小样本筛选，不能跨运行相减、取最快样本或作为正式P95。模式名随原始脚本定义；baseline均为第五阶段合格路径，serial表示分块但不重叠。部分单次stream筛选变快，独立复验和与新MLP组合未形成稳定优势。共享native上下文的组合筛选还会包含动态shape切换成本；这些结果不证明所有流水实现都无效。</p>
{table(['路线','各模式P50 ms（非正式）','原始记录'],negative)}
<p>实际层序为linear、linear、full、linear、linear、full、linear、linear、full、full、full、full。跨层Q/K/V预取只针对1→2、4→5、7→8三处，完整615行K/V齐备后才进入注意力。初版测试误期望5次命中，保存失败记录后按实际目标集合修正，没有改变层序。</p>
<p>固定Toolkit2 2.5.0的显式BF16 Cast图在构建阶段因NumPy类型转换失败，half/round控制可以构建；这仅说明当前构建路径受阻，不代表所有编译器都不支持BF16。减少临时数组的BF16融合通过数值/保护测试但组件更慢。另有组件初版漏暖机一个新对照句柄：整批原样归档为无效性能筛选，修正后全部模式暖机并检查句柄稳定，完整重跑；正式样本不受影响。还发现原长短控制同时保留33个闲置旧MLP对照句柄会触发SDK分配失败；按长度退役无用缓存后，恢复双方预热流程重跑长短切换和全部正式协议。另有正式取样前完整JSON相等的误断言：旧第五阶段原始数据已有约1e-7浮动，修正为≤1e-6的对照身份附加检查并完整重跑，原0.01精度门槛不变。</p>
<p>{' · '.join(link(x) for x in ('logs/cast-build.json','logs/cast-console.log','logs/boundary-unit.json','logs/screen-cross-transition-count-error.json','logs/split-mlp-initial-cold-control.json','logs/formal-baseline-bitwise-assertion-error.json','logs/short-control-extra-bank-allocation-failure.json','logs/stream-unit-import-error.log','logs/native-mlp-unit-weight-path-error.log'))}</p>
<h3>复现与线上保留</h3>
<p>新目录/opt/decision3-edge-bf16-pipeline-20261018，无端口进程，CPU4–7、Torch3线程、原扫描4线程、8GiB限制。线上容器{service['id'][:12]}健康，启动时间/配置及{len(service['files'])}个发布文件SHA不变；五次完整HTTP响应与实验前相同。验收前后{s['artifact_files_checked']}个源代码/库/模型/SDK/参照文件SHA不变，实验进程已退出。未反编译厂商库。</p>
<p>{' · '.join(link(x) for x in ('REPRODUCTION.md','VERIFY.txt','DESIGN.md','logs/artifacts-before.json','logs/artifacts-after.json','service-before.json','service-after.json'))}</p>
</section><div class="box note"><b>以下保留前五阶段历史；最新结论以第六阶段为准。</b></div>'''
subprocess.run([sys.executable,str(root/'fusion-pipeline-experiment/update_report.py')],check=True,stdout=subprocess.DEVNULL)
path=root/'mlp-length-experiment/report.html';text=path.read_text().replace('<nav>','<nav><a href="#bf16-pipeline">第六阶段BF16融合</a>',1).replace('</nav>','</nav>'+section,1).replace('<title>d3-edge 多长度推理与615行剩余瓶颈实验</title>','<title>d3-edge 615 token BF16融合与流水实验</title>');text=text.replace('以下保留前四阶段历史，最新结论以第五阶段为准。','以下保留前四阶段历史；第五阶段为历史快照，当前最新见第六阶段。');path.write_text(text);print('UPDATED',path)
