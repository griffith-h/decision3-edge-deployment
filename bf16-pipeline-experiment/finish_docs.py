"""Reconcile final validated evidence into existing entry points."""
from pathlib import Path
import json,statistics,re
P=Path(__file__).resolve().parent;root=P.parent
s=json.loads((P/'logs/summary.json').read_text());assert s['passed'];b,c=s['baseline'],s['candidate'];profile=json.loads((P/'logs/profile-native.json').read_text());formal=[json.loads((P/f'logs/formal-{i}.json').read_text()) for i in (1,2,3)];m=s['regression_maxima'];ms=lambda x:f'{x*1000:.2f}'
def median(tag,mode='candidate'):return statistics.median(x['components'].get(tag,{}).get('inclusive',0) for x in profile['records'] if x['mode']==mode)
new=f'''## 第六阶段：验收完成，1秒目标未达成

- 最新已测精度合格候选为 [accepted_owner.py](bf16-pipeline-experiment/accepted_owner.py)，[selection.json](bf16-pipeline-experiment/selection.json)记录路径。三轮独立容器、每轮30组暖态交错配对，合并90次候选P50 **{ms(c['median'])}ms**、P95 **{ms(c['p95_nearest_rank'])}ms**；第五阶段合格对照为{ms(b['median'])}ms / {ms(b['p95_nearest_rank'])}ms，P50下降{s['reduction']*100:.2f}%。{s['paired_candidate_wins']}/90组更快。仍未达到P50/P95均<1秒；这是B1/T615模型方法调用，不是HTTP/并发SLA。
- [报告第六阶段](mlp-length-experiment/report.html#bf16-pipeline) · [复现](bf16-pipeline-experiment/REPRODUCTION.md) · [核验](bf16-pipeline-experiment/VERIFY.txt) · [原始汇总](bf16-pipeline-experiment/logs/summary.json)。板端 `/opt/decision3-edge-bf16-pipeline-20261018`，无端口容器 `decision3-bf16-pipeline-test` 已退出；前五阶段只读。未部署；研究成果随本次main提交归档。
- 原BF16权重不变，公开FP16×FP16→FP32 MatMul直接在native C上完成原BF16 gate/up舍入、SiLU和乘法舍入，写入down的native A。两个独立down句柄以公开mask1/2处理320/295行，完整合并全部615行。长度切换前退役闲置半精度/旧FP32 MLP对照缓存，保留native MLP暖态句柄。并非直接扩展旧115行FP16 native布局尝试。
- 34例新推理全部通过原current门槛，相对上一合格候选最大受检数值差{s['max_delta_vs_previous_qualified']:.8f}。相对原current最大Δ概率{m['probability_max_delta']:.8f}、Δscore{m['score_max_delta']:.8f}、Δnoul{m['noul_max_delta']:.8f}；choice一致。confidence独立报告，最大差{m['confidence_max_delta']:.8f}。score范围0–2，不能称为概率差；数值一致性不等于业务准确率。
- 全部中间输出私有，gate/up及down故障都等待全部worker，再整层原CPU MLP重算；运行故障层持续CPU。输入/乘积溢出、有限65496原饱和保护、down有限65536输出保护、范围后恢复、部分创建及重复关闭通过。
- 已试原32/64扫描的128/256/512行消费流水，三处下一全注意力层norm/Q/K/V预取，native投影扩展及与新MLP组合；未稳定胜过最终路径，未采用。实际层序不是严格交替；cross只对1→2、4→5、7→8，完整615 K/V齐备后才做注意力。扫描/卷积状态、12层及非因果mask保持。
- 固定Toolkit2 2.5.0 BF16 Cast构建因NumPy类型错误失败，不能据此推广为所有编译器不支持。减少临时数组融合组件更慢。组件初版漏暖机新对照句柄、cross命中数误断言、导入/权重路径错误及formal完整JSON相等误断言全部保留；性能无效运行整批重跑，不挑掉慢样本。正式对照身份附加检查≤1e-6，用于已存在约1e-7浮动；原0.01精度门槛不变。
- 独立5组profile：注意力{ms(median('attention_entry'))}ms，其他投影{ms(median('projection_rows'))}ms，扫描{ms(median('scan_entry'))}ms（原生{ms(median('scan_c'))}ms），MLP{ms(median('mlp'))}ms。inclusive入口已含子项，各项中位数/重叠worker时间不能相加。SDK区间包含提交/等待/转换，不是纯NPU时间。
- 三轮暖态句柄/FD稳定，关闭后engine/native MLP/attention句柄均0；最高进程RSS约{s['formal_rss_max_kib']/1024/1024:.2f}GiB，不代表全部驱动/NPU内存。验收前后{s['artifact_files_checked']}项资产/参照SHA不变，线上身份/启动时间/配置和{s['production_files_checked']}个发布文件SHA不变，健康、五次完整HTTP输出一致。

'''
f=root/'WORK_STATUS.md';text=f.read_text();start=text.index('## 第六阶段：');end=text.index('## 第五阶段：',start);text=text[:start]+new+text[end:]
text=text.replace('本记录随四轮多长度独立实验一并提交。先看 [最新HTML报告第四阶段](mlp-length-experiment/report.html#long-tail)，','前四轮已提交；第五、第六阶段随本次main提交归档。先看 [最新HTML报告第六阶段](mlp-length-experiment/report.html#bf16-pipeline)，')
text=text.replace('[service-after.json](fusion-pipeline-experiment/service-after.json)','[service-after.json](bf16-pipeline-experiment/service-after.json)').replace('本轮不部署、不推送。','本次提交不部署。').replace('最优已测精度合格候选为 fusion-pipeline-experiment','第五阶段当时最优已测精度合格候选为 fusion-pipeline-experiment')
text=re.sub(r'(?:第五阶段当时)+(?=最优已测精度合格候选为 fusion-pipeline-experiment)', '第五阶段当时', text)
text=text.replace('## 前四阶段历史实验','## 前四阶段历史实验')
text=text.replace('统一报告由 `python3 fusion-pipeline-experiment/update_report.py`','统一报告由 `python3 bf16-pipeline-experiment/update_report.py`').replace('之后五轮以本报告','之后六轮以本报告')
start=text.index('## 剩余瓶颈与继续条件');end=text.index('## 复现、检查与资产位置',start)
without=statistics.median(x['seconds']-x['components']['attention_entry']['inclusive'] for x in profile['records'] if x['mode']=='candidate')
text=text[:start]+f'''## 剩余瓶颈与继续条件

最新候选的成本仍分布在大型投影、完整扫描、原BF16边界MLP及CPU准备/残差。仅假设整个注意力入口为零，独立profile其余时间中位数仍约{ms(without)}ms；这不是硬件理论下限。P50还需削减约{max(0,c['median']-1)*1000:.0f}ms，并同时压低P95。当前已测路线没有支持进入1秒的证据。

保留BF16边界的native C→A融合和并行down已落实。现有分段流水、三处下一全注意力层预取及native投影扩展没有稳定胜过最新合格路径；不能把单次小样本改善或组件理论重叠直接相加。未来更深的跨层扫描/编译路径仍须先证明公开接口、完整FP32状态/卷积边界、数值和资源闭合，再做端到端；未实现机制不是本轮已验证能力。不能回用失败FP16图、放宽0.01门槛、按内容挑选路径或承诺200ms。结论限当前SDK/频率/四CPU及已测实现，不是RK3576硬件下限。

'''+text[end:]
text=text.replace('板端最新目录 /opt/decision3-edge-fusion-pipeline-20261018；前四轮目录只读。run4.sh固定','板端最新目录 /opt/decision3-edge-bf16-pipeline-20261018；前五轮目录只读。run.sh固定')
old='- 最新核验含34例原current回归、三轮各30组正式对照、115行12组、故障与关闭单测、保存记录重算、252项板端运行资产及线上554个发布文件SHA。1001条保存记录比较含归档副本，排除相同副本后859条，不是独立推理次数。'
text=text.replace(old,f'- 第六阶段核验含34例原current回归、三轮各30组正式对照、115行12组、独立profile、故障/范围/关闭单测、保存记录重算、{s["artifact_files_checked"]}项资产及线上554个发布文件SHA。第五阶段1001条比较含归档副本，去重859条，仅是历史保存记录核对，不算本轮新推理。')
text=text.replace('python3 fusion-pipeline-experiment/validate_records.py\npython3 fusion-pipeline-experiment/check_report.py','python3 bf16-pipeline-experiment/validate_records.py\npython3 bf16-pipeline-experiment/check_report.py').replace('python3 -m compileall -q fusion-pipeline-experiment','python3 -m compileall -q bf16-pipeline-experiment fusion-pipeline-experiment').replace('for f in fusion-pipeline-experiment/*.sh','for f in bf16-pipeline-experiment/*.sh')
f.write_text(text)
f=root/'README.md';text=f.read_text().replace('五轮未部署实验','六轮未部署实验');lines=text.splitlines()
for i,line in enumerate(lines):
 if line.startswith('[多长度推理 HTML 报告]'):
  lines[i]=f'[多长度推理 HTML 报告](mlp-length-experiment/report.html#bf16-pipeline) 保留六轮实验。最新615 token暖态方法：上一合格候选P50 {ms(b["median"])}ms → {ms(c["median"])}ms，候选P95 {ms(c["p95_nearest_rank"])}ms，P50下降{s["reduction"]*100:.2f}%；1秒目标未达成。34例原current数值门槛通过，confidence独立报告；不代表业务准确率。未部署；研究成果随本次main提交归档，适用范围和原始记录见第六阶段及 [WORK_STATUS.md](WORK_STATUS.md)。'
f.write_text('\n'.join(lines)+'\n')
repro=P/'REPRODUCTION.md';text=repro.read_text().replace('（进行中）','（验收完成，1秒未达成）')
marker='\n## 最终采用路径与验收\n'
if marker in text:text=text.split(marker)[0]
text+=f'''{marker}
采用`accepted_owner.Owner`（`split_owner.py`），由`selection.json`记录：仅B1/T615 logits，公开未量化FP16×FP16→FP32、native C→A原BF16边界融合；down以mask1/2处理320/295行，原次序拼回615行。其他投影沿用第五阶段normal FP32，扫描保持原32/64、4线程，流水和预取未采用。上下文分别持有独立A/B/C缓冲，全部归新bank所有；旧engine/pool只借用，不能用旧mixed ABI销毁native bank。

```bash
bash build-selected.sh
bash run.sh test_native_ranges.py
# 回到本机：python3 bf16-pipeline-experiment/capture_artifacts.py before
bash run.sh regression_native.py
bash run.sh formal_native.py --round 1 --pairs 30
bash run.sh formal_native.py --round 2 --pairs 30
bash run.sh formal_native.py --round 3 --pairs 30
bash run.sh short_formal.py --round 1 --pairs 12
bash run.sh profile_native.py
# 全部退出后在本机：
python3 bf16-pipeline-experiment/capture_artifacts.py after
python3 bf16-pipeline-experiment/preserve_service.py after
python3 bf16-pipeline-experiment/validate_records.py
python3 bf16-pipeline-experiment/update_report.py
python3 bf16-pipeline-experiment/check_report.py
```

正式90次候选P50 {ms(c['median'])}ms、P95 {ms(c['p95_nearest_rank'])}ms，原34例门槛通过，相对上一合格候选最大受检数值差{s['max_delta_vs_previous_qualified']:.8f}。原current参照来自只读第五阶段34份独立current，验收哈希包含参照文件；本轮另新算615 current并断言相等。输出没有作为推理缓存使用。范围/故障/关闭详见VERIFY和原始JSON。

正式前完整JSON相等误断言保存为`formal-baseline-bitwise-assertion-error.*`，没有产生正式样本。旧第五阶段回归与正式输出已有约1e-7浮动；身份附加检查改为全部数值≤1e-6、choice一致，实际完整baseline输出和delta先保存后断言。原current的0.01概率/score/noul门槛从未放宽，confidence仍独立报告。所有正式样本保留；组件漏暖机初版归档`split-mlp-initial-cold-control.*`，不能混入暖态指标。

SHA扫描仅在测试停止后运行；失败身份断言前后323项资产相同，保存为`artifacts-baseline-assertion-aborted-*`。重新冻结后最终{s['artifact_files_checked']}项验收资产前后相同。各运行的`logs/sources/`保存当时源码；本地报告/校验后处理可能晚于数据采集，不能替代实际运行快照。未覆盖多Owner并发、冷启动SLA、任意业务输入或业务准确率。
'''
repro.write_text(text)
verify=f'''第六阶段验收结果（615 token暖态方法；未部署；研究成果随本次main提交归档）

目标P50/P95均<1秒：未达到。
三轮独立容器×30组暖态交错配对，候选90次：P50 {ms(c['median'])}ms，nearest-rank P95 {ms(c['p95_nearest_rank'])}ms。
同轮第五阶段合格对照：P50 {ms(b['median'])}ms，P95 {ms(b['p95_nearest_rank'])}ms；P50下降{s['reduction']*100:.2f}%，{s['paired_candidate_wins']}/90组更快；所有慢样本保留。

原current精度：regression_native.py，34例新输出全部通过，choice一致；与上一合格候选最大受检数值差{s['max_delta_vs_previous_qualified']:.8f}。
最大分类概率差{m['probability_max_delta']:.10f}，score差{m['score_max_delta']:.10f}，noul差{m['noul_max_delta']:.10f}，原门槛各≤0.01。
confidence差{m['confidence_max_delta']:.10f}独立报告。score范围0–2，score差不是概率差；数值一致性不等于业务准确率。

已执行板端命令与结果（详见同名logs/*.json/.log和logs/sources快照）：
- build.sh/build-native.sh/build-native-split.sh/build-selected.sh：公开SDK构建exit0，-Wall/-Wextra有既有单行格式告警，未修改厂商库。
- run.sh test_stream.py：18例原32/64扫描逐位一致，含615尾块；导入错误初版保留。
- run.sh test_boundary.py：BF16边界、NaN/Inf/饱和及乘积保护通过；减少临时缓冲组件未更快。
- run.sh probe_native_attrs.py：公开实际FP32 C=(N/4,M,4)、A=(K/8,M,8)，动态shape逐项验证。
- run.sh test_native_mlp.py：真实0/5/11层615激活，正常FP32与native MLP逐位一致，12组配对/层。
- run.sh test_split_mlp.py：全部模式暖机、句柄稳定后每层12组轮转，normal/native/split逐位一致；初版冷句柄污染整批归档，不作性能证据。
- run.sh test_native_guards.py / test_split_guards.py：gate/up及down故障等待、完整CPU重算、持续CPU、范围恢复、组投影peer等待、部分创建及重复关闭通过。
- run.sh test_native_ranges.py：有限65496原饱和保护-3、有限down 65536输出保护-3、输入NaN/Inf/溢出-2、有限乘积溢出-2和后续恢复通过。
- run.sh test_stream_guards.py：生产者/第二块消费者故障整层重算，全部线程join，故障层持续旁路、短输入及重复关闭通过。
- run.sh screen_pipeline.py（128/256/512）/screen_cross.py/screen_native_projection.py/screen_native_stream.py/screen_split.py/screen_split_stream.py：全部完成；筛选不是正式P95，全部计时输出保存；部分暖机只保存检查结果。跨层实际3处，误期望5的断言保留。
- run.sh test_cache.py / test_cache_integration.py：所有权保护、私有packed模块登记、33个旧MLP对照句柄及半精度缓存退役、615/115/615与多batch切换、重复关闭通过。
- run.sh logs/postlude/test_bank_lifecycle.py：真实Owner关闭及GC的FD清单、三轮独立bank/worker池构造释放无FD增长。
- run.sh regression_native.py：34例通过，新增路径只命中13个B1/T615请求，每次11层MLP；其他形状0次。
- run.sh formal_native.py --round 1/2/3 --pairs 30：共180个模式输出，90份候选；句柄/FD暖态稳定，关闭后engine/native MLP/attention全0；设备/CPU识别/perf-map等运行时FD仍可能保留，独立三轮bank/worker池循环无增长。
- run.sh short_formal.py --round 1 --pairs 12：24份115 token输出，新路径命中0。
- run.sh profile_native.py：独立5组插桩，10份输出，全数值门槛通过；worker区间重叠，SDK时间不是纯NPU时间。

协议修复：旧短控保留额外33个旧MLP对照句柄触发SDK分配失败；按长度退役无用缓存后，用原双方预热流程重跑。旧三轮数据原样保留在pre-cache-policy，不与最终数据混合。正式取样前完整JSON相等误断言没有产生有效计时；旧第五阶段记录已存在约1e-7浮动。附加对照身份检查≤1e-6，原0.01门槛不变；重跑完整正式协议。失败阶段323项资产SHA相同。最终{s['artifact_files_checked']}项资产/参照SHA前后一致。
线上：容器身份/启动时间/配置、554个发布文件SHA不变，健康，5次完整HTTP响应一致；实验进程退出。最高正式RSS约{s['formal_rss_max_kib']/1024/1024:.2f}GiB，不代表所有驱动内存。

本地静态/保存记录验证（不重跑板端推理）：
python3 bf16-pipeline-experiment/validate_records.py
python3 bf16-pipeline-experiment/update_report.py
python3 bf16-pipeline-experiment/check_report.py
python3 -m compileall -q bf16-pipeline-experiment
for f in bf16-pipeline-experiment/*.sh; do bash -n "$f"; done
git diff --check

结论范围：固定SDK/频率、4个CPU、单Owner顺序方法调用、上述34例；不是硬件下限、冷启动/HTTP并发SLA或业务准确率保证。没有足以同时证明1秒P50/P95的已测路线。
'''
(P/'VERIFY.txt').write_text(verify);print('DOCS UPDATED')
