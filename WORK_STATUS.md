# 继续工作入口

本记录随四轮多长度独立实验一并提交。先看 [最新HTML报告第四阶段](mlp-length-experiment/report.html#long-tail)，再按对应复现说明操作；所有数据为现有RK3576实测快照，不代表任意机器或业务输入的保证。

## 线上服务与约束

- 线上仍是 `decision3-edge-graph`，`http://192.168.1.164:8900/v1/systemone`；graph-half v1，快路径适用于batch1、115 token，其余长度保留原回退。
- 冻结发布 `/opt/decision3-edge-graph-20261018-v1`；回滚流程见 [已部署版本说明](cpu-tail-deployment/README.md)。不要将后续实验脚本当成部署脚本。
- 最近一次保留核验见 [service-after.json](long-tail-experiment/service-after.json)：容器ID前缀 `07577e5f9827`、启动时间、Config/HostConfig与554个发布文件SHA不变，健康，5次完整HTTP输出与实验前一致。独立实验容器已退出。此处是核验时状态，后续操作前应重新检查。
- 继续优化时保持模型权重、原12层、非因果完整注意力和扫描状态语义；不反编译厂商库，保留线上，在无端口独立进程里测试。本次提交不部署。

## 已完成四轮实验

| 顺序 | 目录及复现入口 | 完成内容 |
| --- | --- | --- |
| 1 | [mlp-length-experiment](mlp-length-experiment/REPRODUCTION.md) | 复用115行图，新增128/384行图并按实测成本组合，MLP内部展平及补零；记录不同长度收益和误差。 |
| 2 | [row-projection-experiment](row-projection-experiment/REPRODUCTION.md) | 42个其他大型投影按行分块，单次native最多512行；先拼完整输出再做原注意力/扫描。保留严格score保护版与快版对照。 |
| 3 | [long-sequence-experiment](long-sequence-experiment/REPRODUCTION.md) | 自有CPU完整注意力和GatedDeltaNet扩展到512<T≤2048，6层完整注意力、6层扫描命中；完整模型范围以正式JSON为准。 |
| 4 | [long-tail-experiment](long-tail-experiment/REPRODUCTION.md) | 分解615行2.4秒；比较12注意力+29扫描配置，采用更快的C/NEON检查及已有MLP图双句柄调度。 |

统一报告由 `python3 long-tail-experiment/update_report.py` 重建，按顺序调用旧阶段生成器，保留原始数据与失败路线。不要只运行第一阶段生成器覆盖最新报告。统一全过程历史 `d3-edge-development-history.html` 截止graph-half发布；之后四轮以本报告及各目录为准。

## 当前最优独立候选与边界

- 第四轮候选模式 `validate_mlp`，源码入口 `long-tail-experiment/runner.py`。保留原完整注意力与原扫描32行块/64列tile计算，仅替换检查与MLP调度。
- 615行20组交错配对：2441.3780ms → 2153.2309ms，下降11.8026%；候选P95 2189.1169ms。这是暖态模型方法延迟，不是HTTP/并发SLA。
- 独立5组组件profile：注意力约745ms、扫描383ms（原生计算305ms）、其他大型投影387ms、MLP303ms；分项中位数不要求相加等于总中位数。
- 新MLP调度只命中 `[1,615,1024]` 且原计划为128+128+384。原128句柄串行两次，384句柄独立并行；每层仍3次调用、补25行。公开核心掩码0=AUTO、1=core0、2=core1。其他形状保持旧调度，不能直接推广。
- 615行仅调检查的收益和其他长输入回归见正式JSON。注意力query分块全模型收益不足2%未采用；扫描块/线程/直接递推未胜出。全部配置和失败日志保留。
- 58份正式候选加24份额外短请求，相对上一阶段baseline全部受检数值差0。615行current新算并与第三轮冻结参照相等；其他正式场景明确复用第三轮current，额外24份仅对baseline，不能算独立current对照。
- 原门槛为choice一致、概率/score/noul绝对差≤0.01，confidence另列。multi-wide（T239、B×T717）仍有原MLP组合导致的score约0.01745累计偏差，本轮未修复；615行累计概率差0.00479828、confidence差0.01403546。本轮零增量不代表对原模型完全无差或业务准确率无损。

## 继续优化的候选方向

1. 针对完整注意力约745ms，研究自有CPU QK/softmax/PV更深融合与中间搬运，或公开RKNN分块注意力图；必须保留全K/V、GQA、mask与归一化。此前115行RKNN注意力图精度失败见 [R17](d3-edge-development-history.html#R17)，不能只扩shape。
2. 细分约387ms其他投影的提交、等待和转换，验证独立工作流水能否减少耗时，并核对NPU争用。
3. 若推广MLP调度，按实际长度、batch和计划单独测速度/输出；不要将615专用策略默认套到其他形状。
4. 扫描约305ms计算需要新的实质实现路径；已测块大小/线程/递推没有明显收益。

这些均未验证。当前没有1秒或200ms的615行结果；仅消除注意力，其余部分的profile中位数仍约1.37秒。这是当前执行分解，不是硬件理论下限。

## 复现、检查与资产位置

- 板端新实验目录 `/opt/decision3-edge-long-tail-20261018`，前置三个目录同名。`run.sh`将冻结发布、旧实验及模型只读挂载；固定CPU4–7、三线程、8GiB，不发布端口，板端任务顺序执行。
- 原模型快照 `20c745653400d191ecacd99c4eafc0c5900dc15e`。模型权重、生成的`.rknn`、`.so`、捕获激活`.pt`和SDK下载内容按既有规则不入Git。板端/本地工作区仍保留现有产物，换机器按各阶段说明重建；克隆仓库不会自动得到可运行镜像或编译图。
- 各阶段 `logs/` 保留完整输出、配对计时、选型、边界单测、错误日志、源码快照及SHA；`VERIFY.txt`列明已执行命令和结果。
- 最近一轮已通过聚焦单测、738条native回放记录检查、82份候选输出核对、保存输出误差重算、线上保留及源码/依赖SHA检查。未验证服务并发、任意输入或业务准确率。

本地只读复核（不运行板端推理、不改服务）：

```bash
python3 long-tail-experiment/validate_records.py
python3 long-tail-experiment/check_report.py
python3 -m compileall -q mlp-length-experiment row-projection-experiment long-sequence-experiment long-tail-experiment
bash -n mlp-length-experiment/run.sh row-projection-experiment/run.sh long-sequence-experiment/run.sh long-tail-experiment/run.sh long-tail-experiment/build.sh
git diff --check
```

重跑板端或更新文档时，先阅读对应 `REPRODUCTION.md`；哈希扫描不与计时/HTTP重叠，不在测试运行中覆盖源码/库。四轮中“本阶段未提交/推送”的描述是当时实验状态，本次统一提交不改写历史日志。
