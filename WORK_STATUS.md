# 继续工作入口

前四轮已提交；第五、第六阶段随本次main提交归档。先看 [最新HTML报告第六阶段](mlp-length-experiment/report.html#bf16-pipeline)，再按对应复现说明操作；所有数据为现有RK3576实测快照，不代表任意机器或业务输入的保证。

## 线上服务与约束

- 线上仍是 `decision3-edge-graph`，`http://192.168.1.164:8900/v1/systemone`；graph-half v1，快路径适用于batch1、115 token，其余长度保留原回退。
- 冻结发布 `/opt/decision3-edge-graph-20261018-v1`；回滚流程见 [已部署版本说明](cpu-tail-deployment/README.md)。不要将后续实验脚本当成部署脚本。
- 最近一次保留核验见 [service-after.json](bf16-pipeline-experiment/service-after.json)：容器ID前缀 `07577e5f9827`、启动时间、Config/HostConfig与554个发布文件SHA不变，健康，5次完整HTTP输出与实验前一致。独立实验容器已退出。此处是核验时状态，后续操作前应重新检查。
- 继续优化时保持模型权重、原12层、非因果完整注意力和扫描状态语义；不反编译厂商库，保留线上，在无端口独立进程里测试。本次提交不部署。

## 本次main提交汇总

- 归档第五阶段`fusion-pipeline-experiment`与第六阶段`bf16-pipeline-experiment`的自有源码、配置、复现说明、正式原始JSON、失败日志和运行源码快照；二进制、模型权重及厂商SDK继续按既有规则忽略。
- 第五阶段采用动态完整注意力融合、615行投影及精度修复，正式90次候选P50/P95为1745.16/1776.55ms；修复既有multi-wide score超门槛问题。第六阶段采用保留BF16边界的native MLP融合、320/295行并行down与长度域缓存退役，另一次独立90组对照为1743.55/1770.60ms → 1579.62/1642.71ms，P50下降9.40%。不同轮次的统计不混合。
- 1秒目标未达成。最新34例原current门槛通过，confidence独立报告；故障回退、长度切换、资源关闭及线上保留核验完成。跨层流水、投影扩展及其他未胜出路线保留负面证据。详细适用条件和命令见下方各阶段入口。
- 本次仅发布Git研究记录，线上仍为graph-half v1；不部署。已有记录中的“未提交/未推送”描述实验验收时状态，原始日志和哈希不回写。

## 第六阶段：验收完成，1秒目标未达成

- 最新已测精度合格候选为 [accepted_owner.py](bf16-pipeline-experiment/accepted_owner.py)，[selection.json](bf16-pipeline-experiment/selection.json)记录路径。三轮独立容器、每轮30组暖态交错配对，合并90次候选P50 **1579.62ms**、P95 **1642.71ms**；第五阶段合格对照为1743.55ms / 1770.60ms，P50下降9.40%。90/90组更快。仍未达到P50/P95均<1秒；这是B1/T615模型方法调用，不是HTTP/并发SLA。
- [报告第六阶段](mlp-length-experiment/report.html#bf16-pipeline) · [复现](bf16-pipeline-experiment/REPRODUCTION.md) · [核验](bf16-pipeline-experiment/VERIFY.txt) · [原始汇总](bf16-pipeline-experiment/logs/summary.json)。板端 `/opt/decision3-edge-bf16-pipeline-20261018`，无端口容器 `decision3-bf16-pipeline-test` 已退出；前五阶段只读。未部署；研究成果随本次main提交归档。
- 原BF16权重不变，公开FP16×FP16→FP32 MatMul直接在native C上完成原BF16 gate/up舍入、SiLU和乘法舍入，写入down的native A。两个独立down句柄以公开mask1/2处理320/295行，完整合并全部615行。长度切换前退役闲置半精度/旧FP32 MLP对照缓存，保留native MLP暖态句柄。并非直接扩展旧115行FP16 native布局尝试。
- 34例新推理全部通过原current门槛，相对上一合格候选最大受检数值差0.00000047。相对原current最大Δ概率0.00929520、Δscore0.00705960、Δnoul0.00301053；choice一致。confidence独立报告，最大差0.01414562。score范围0–2，不能称为概率差；数值一致性不等于业务准确率。
- 全部中间输出私有，gate/up及down故障都等待全部worker，再整层原CPU MLP重算；运行故障层持续CPU。输入/乘积溢出、有限65496原饱和保护、down有限65536输出保护、范围后恢复、部分创建及重复关闭通过。
- 已试原32/64扫描的128/256/512行消费流水，三处下一全注意力层norm/Q/K/V预取，native投影扩展及与新MLP组合；未稳定胜过最终路径，未采用。实际层序不是严格交替；cross只对1→2、4→5、7→8，完整615 K/V齐备后才做注意力。扫描/卷积状态、12层及非因果mask保持。
- 固定Toolkit2 2.5.0 BF16 Cast构建因NumPy类型错误失败，不能据此推广为所有编译器不支持。减少临时数组融合组件更慢。组件初版漏暖机新对照句柄、cross命中数误断言、导入/权重路径错误及formal完整JSON相等误断言全部保留；性能无效运行整批重跑，不挑掉慢样本。正式对照身份附加检查≤1e-6，用于已存在约1e-7浮动；原0.01精度门槛不变。
- 独立5组profile：注意力229.72ms，其他投影387.32ms，扫描342.38ms（原生263.12ms），MLP354.36ms。inclusive入口已含子项，各项中位数/重叠worker时间不能相加。SDK区间包含提交/等待/转换，不是纯NPU时间。
- 三轮暖态句柄/FD稳定，关闭后engine/native MLP/attention句柄均0；最高进程RSS约6.49GiB，不代表全部驱动/NPU内存。验收前后329项资产/参照SHA不变，线上身份/启动时间/配置和554个发布文件SHA不变，健康、五次完整HTTP输出一致。

## 第五阶段：验收完成，1秒目标未达成

- 第五阶段当时最优已测精度合格候选为 fusion-pipeline-experiment/selected_owner.py，冻结配置为 execution.json。三轮独立无端口容器、每轮30组交错配对，合并90份候选P50 **1745.16ms**、P95 **1776.55ms**；固定validate_mlp P50 2123.25ms，下降17.81%。这是B1/T615暖态方法耗时，不是HTTP/并发SLA，低于1秒的目标没有达到。
- [报告第五阶段](mlp-length-experiment/report.html#fusion-pipeline) · [复现](fusion-pipeline-experiment/REPRODUCTION.md) · [核验命令与结果](fusion-pipeline-experiment/VERIFY.txt) · [原始汇总](fusion-pipeline-experiment/logs/summary.json)。板端 /opt/decision3-edge-fusion-pipeline-20261018，独立容器 decision3-fusion-pipeline-test 已退出；线上未替换，本轮未提交、未推送。
- 采用公开动态QK/PV MatMul的FP32输出、自有准备/FP32 softmax/转换融合；大型投影615行一次提交（内部640补零）。原32行块/64列tile扫描仅使用4线程；Torch仍3、CPU4–7。全部真实K/V、非因果mask、权重、12层及六层扫描跨块状态保留。
- 最终615资格恢复FP32投影/MLP输出和原BF16 MLP中间边界，FP16/FP32句柄银行分离。历史R14已使用过FP32输出，本轮用于修复新615内容的双重舍入；不是新硬件能力。其他形状沿用原资格路径，短多batch另保留精度修复。
- 私有D3.logits末层优化：前11层全部615位置，第12层K/V仍615，query/逐行MLP/残差/Norm只算消费者使用的末行。公开encoder仍返回完整hidden，末行MLP使用原CPU及BF16边界。结构/权重版本/方法/钩子/模式改变旁路；错误丢弃私有结果，整次原logits重算并持续禁用。没有截断输入。
- 34例相对单独新算original current全部通过原门槛，包含13份不同内容615输入及12份额外115输入。最大Δ概率0.00929520、Δscore0.00705989、Δnoul0.00301053；choice一致。confidence独立报告，最大差0.01414562，不纳入原0.01门槛。multi-wide score由既有约0.01745降为0.00269004。数值一致性不等于业务准确率。
- 更快FP16组合与缩减FP32范围在新615内容上超门槛，未采用。新640 MLP图组合消融未胜出；staging、投影拆列、双head合并、融合扫描、NEON预测/状态核、融合卷积等负面证据保留。早期动态B未同步、K624对齐、隐藏allclose和测试脚本失败均有原日志/源码，不把失败包装为硬件极限。
- 独立五组profile：注意力226.41ms，其他投影401.94ms，扫描336.10ms（原生254.21ms），MLP505.24ms（原边界准备63.84ms为嵌套子项）。各项中位数和重叠worker时间不能相加。假设整个注意力为零，其余仍约1532.27ms；仅继续优化注意力不足以到1秒。这不是未来实现的硬件下限。
- 115行另做12组：baseline P50 326.30ms、候选325.68ms；全部慢样本保留，不计入615目标。
- 最终故障注入、模式旁路、持续禁用、重复关闭通过。三轮均106个engine句柄和1037个FD暖态稳定，自有MatMul关闭后为0；RSS高水位约6.33GiB，不代表全部驱动内存。板端252项运行资产SHA前后相同；线上身份/启动时间/配置和554个发布文件SHA不变，健康、五次完整HTTP响应一致。未测多Owner并发、任意输入或业务准确率。

## 已完成四轮实验

| 顺序 | 目录及复现入口 | 完成内容 |
| --- | --- | --- |
| 1 | [mlp-length-experiment](mlp-length-experiment/REPRODUCTION.md) | 复用115行图，新增128/384行图并按实测成本组合，MLP内部展平及补零；记录不同长度收益和误差。 |
| 2 | [row-projection-experiment](row-projection-experiment/REPRODUCTION.md) | 42个其他大型投影按行分块，单次native最多512行；先拼完整输出再做原注意力/扫描。保留严格score保护版与快版对照。 |
| 3 | [long-sequence-experiment](long-sequence-experiment/REPRODUCTION.md) | 自有CPU完整注意力和GatedDeltaNet扩展到512<T≤2048，6层完整注意力、6层扫描命中；完整模型范围以正式JSON为准。 |
| 4 | [long-tail-experiment](long-tail-experiment/REPRODUCTION.md) | 分解615行2.4秒；比较12注意力+29扫描配置，采用更快的C/NEON检查及已有MLP图双句柄调度。 |

统一报告由 `python3 bf16-pipeline-experiment/update_report.py` 重建，按顺序调用旧阶段生成器，保留原始数据与失败路线。不要只运行第一阶段生成器覆盖最新报告。统一全过程历史 `d3-edge-development-history.html` 截止graph-half发布；之后六轮以本报告及各目录为准。

## 第四阶段历史参照与旧边界

- 第四轮候选模式 `validate_mlp`，源码入口 `long-tail-experiment/runner.py`。保留原完整注意力与原扫描32行块/64列tile计算，仅替换检查与MLP调度。
- 615行20组交错配对：2441.3780ms → 2153.2309ms，下降11.8026%；候选P95 2189.1169ms。这是暖态模型方法延迟，不是HTTP/并发SLA。
- 独立5组组件profile：注意力约745ms、扫描383ms（原生计算305ms）、其他大型投影387ms、MLP303ms；分项中位数不要求相加等于总中位数。
- 新MLP调度只命中 `[1,615,1024]` 且原计划为128+128+384。原128句柄串行两次，384句柄独立并行；每层仍3次调用、补25行。公开核心掩码0=AUTO、1=core0、2=core1。其他形状保持旧调度，不能直接推广。
- 615行仅调检查的收益和其他长输入回归见正式JSON。注意力query分块全模型收益不足2%未采用；扫描块/线程/直接递推未胜出。全部配置和失败日志保留。
- 58份正式候选加24份额外短请求，相对上一阶段baseline全部受检数值差0。615行current新算并与第三轮冻结参照相等；其他正式场景明确复用第三轮current，额外24份仅对baseline，不能算独立current对照。
- 原门槛为choice一致、概率/score/noul绝对差≤0.01，confidence另列。multi-wide（T239、B×T717）仍有原MLP组合导致的score约0.01745累计偏差，本轮未修复；615行累计概率差0.00479828、confidence差0.01403546。本轮零增量不代表对原模型完全无差或业务准确率无损。

## 剩余瓶颈与继续条件

最新候选的成本仍分布在大型投影、完整扫描、原BF16边界MLP及CPU准备/残差。仅假设整个注意力入口为零，独立profile其余时间中位数仍约1360.22ms；这不是硬件理论下限。P50还需削减约580ms，并同时压低P95。当前已测路线没有支持进入1秒的证据。

保留BF16边界的native C→A融合和并行down已落实。现有分段流水、三处下一全注意力层预取及native投影扩展没有稳定胜过最新合格路径；不能把单次小样本改善或组件理论重叠直接相加。未来更深的跨层扫描/编译路径仍须先证明公开接口、完整FP32状态/卷积边界、数值和资源闭合，再做端到端；未实现机制不是本轮已验证能力。不能回用失败FP16图、放宽0.01门槛、按内容挑选路径或承诺200ms。结论限当前SDK/频率/四CPU及已测实现，不是RK3576硬件下限。

## 复现、检查与资产位置

- 板端最新目录 /opt/decision3-edge-bf16-pipeline-20261018；前五轮目录只读。run.sh固定CPU4–7、Torch三线程、扫描四线程、8GiB，无端口，板端任务顺序执行。
- 原模型快照 `20c745653400d191ecacd99c4eafc0c5900dc15e`。模型权重、生成的`.rknn`、`.so`、捕获激活`.pt`和SDK下载内容按既有规则不入Git。板端/本地工作区仍保留现有产物，换机器按各阶段说明重建；克隆仓库不会自动得到可运行镜像或编译图。
- 各阶段 `logs/` 保留完整输出、配对计时、选型、边界单测、错误日志、源码快照及SHA；`VERIFY.txt`列明已执行命令和结果。
- 第六阶段核验含34例原current回归、三轮各30组正式对照、115行12组、独立profile、故障/范围/关闭单测、保存记录重算、329项资产及线上554个发布文件SHA。第五阶段1001条比较含归档副本，去重859条，仅是历史保存记录核对，不算本轮新推理。

本地只读复核（不运行板端推理、不改服务）：

```bash
python3 bf16-pipeline-experiment/validate_records.py
python3 bf16-pipeline-experiment/check_report.py
python3 long-tail-experiment/validate_records.py
python3 -m compileall -q bf16-pipeline-experiment fusion-pipeline-experiment mlp-length-experiment row-projection-experiment long-sequence-experiment long-tail-experiment
for f in bf16-pipeline-experiment/*.sh; do bash -n "$f"; done
git diff --check
```

重跑板端或更新文档时，先阅读对应REPRODUCTION；哈希扫描不与计时/HTTP重叠，不在测试运行中覆盖源码/库。历史中“本阶段未提交/推送”是当时状态；第五、第六阶段研究成果随本次main提交归档，不改写既有原始日志。
