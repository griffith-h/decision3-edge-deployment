# RK3576 上的 d3-edge 部署与 d3.1-nano 实验

本仓库记录 d3-edge 服务的部署、RKNN 推理优化，以及 d3.1-nano 在同一块 RK3576 上的 CPU 和 NPU 实测。

d3.1-nano 的八核模式已在 16 GB RK3576 上用 TypeSafe 和 vLLM 的公开请求实测。55 题发票的热态耗时从旧版同板复测的 20 分 19 秒降至 **11 分 20 秒**，55 个字段与旧版判断一致。12 条有纯 CPU 对照的请求中，58 个字段判断全部一致，170/172 个选项的概率差在 1 个百分点内。精度优先模式在相同对照上通过 172/172 个选项。[八核实测与资源占用](d3-engine/EIGHT-CORE.md) · [引擎与逐项概率](d3-engine/README.md)

截至 2026-11-20，板端连接中断后的四核对照仍待复测。CPU 与 NPU 同时计算一段注意力、合并投影和 Mali 矩阵乘的实验源码已单独归档。现有结果可在普通电脑上用 `python3 -B d3-engine/verify_results.py` 离线复核。[项目状态](WORK_STATUS.md) · [三个原型与复现方法](d3-engine/experiments/long-prefix/README.md)

## d3.1-nano 单问题板测（2026-11-19）

在 16 GB NanoPi R76S 上，同一条 115-token 中文分类请求的热态中位数从原版纯 CPU 的 **108.09 秒**降到默认前向加 NPU 的 **10.87 秒**。单问题一次前向配合 NPU 和 edge 的 CPU 算子后为 **1.53 秒（70.68×）**。三种方式均选中“编程任务”，最快方案相对原版纯 CPU 的最大选项概率差为 **0.004984 个百分点**。[计时、选项概率与复测方法](d3.1-nano-npu-test/README.md)

## 技术总结文章

[我是怎样用 AI，把推理从 6～7 秒推进到 300ms～2s 的](d3-edge-inference-engine-optimization.md)：AI 协作过程复盘，只讲每轮用了什么、怎么推进和取得什么效果，保留 25 条当时发送给 Codex 的指令，部分来自 Codex 建议。已部署短输入与未部署长输入实验分别标注。

[图解 HTML 版](d3-edge-inference-engine-optimization.html)：结果图、优化过程和已发送指令卡片，不展开底层代码教学。[Prompt 原文归档](ai-optimization-prompts.json) 保留实际发送的文字，机器地址已脱敏。`python3 build_optimization_article.py` 从 Prompt 归档、叙述配置和结果 JSON 重建 HTML，`--check` 检查一致性；显式加 `--write-markdown` 才会同时重建 Markdown 草稿，普通重建不会覆盖手动编辑的 Markdown。样式源文件为 `optimization-article.css`，生成的 HTML 内嵌样式，可离线阅读。不运行模型或更改服务。

[文章测试来源与复现说明](article-reproduction.md)：本次使用自建固定请求与回归集，未重跑官方完整准确率基准。短输入 100 次、长输入 90 次均为重复同一请求的性能计时；34 例是输出回归集。说明中提供请求原文、原始 JSON、统计重算与设备复测入口，并列明新机器所需依赖。

## d3-edge 官方来源样本与长输入实验（2026-10）

d3-edge 的 Decision Index 评测冻结了 1,000 条请求，覆盖 831 个题组、43 项基准和 8,508 个问题。整批评测暂停。开发阶段 r080 在独立 CPU 和 native 参照下通过全部 34 例回归，最大非 confidence 差为 0.00943618。完整 18 题请求耗时 270.82 秒，通过数值门槛，仍高于 120 秒目标。后续候选的组件、整请求和拒绝记录可按轮次查阅。[样本、权限与重建](official-evaluation/README.md) · [评测计划](official-evaluation/PLAN.md) · [十月最终结果](official-long-input-experiment/CURRENT.md) · [开发记录](official-long-input-experiment/README.md)

[31秒请求耗时账本](official-long-input-experiment/REQUEST_COST.html)：3组固定请求对照、27项不重复耗时，CPU修复/读取、扫描、注意力合计约77.91%；无插桩中位30.99秒，数值门槛不变。详见[Markdown与复现](official-long-input-experiment/REQUEST_COST.md)，收益估算不代表已实现提速。

账本后的首项选点重算实验r088：8/16独立输出交错均未通过组件性能筛选，已及时停止，未跑候选整请求及34/18题；数值门槛未变，详见[负面结果归档](official-long-input-experiment/rejected-selected-dot-ilp/README.md)。

[选点重算为什么变慢](official-long-input-experiment/SELECTED_DOT_DIAGNOSIS.html)：1040个PMU窗口与行步幅×对齐控制，定位L1回填放大和16路尾部重复计算；未发现K热循环落栈。只诊断，未部署；没有将组件收益当整请求提速。

原 4 路加缓存布局的三组整请求净节省分别为 +0.469、−0.077 和 −1.207 秒，含准备及释放，再扣除安装成本。中位数 −0.077 秒，未达到节省 1 秒的条件。81 项 ARM 检查通过，响应和修复计数一致，缓存活跃峰值 63.656 MiB。候选到此停止。[完整请求实测](official-long-input-experiment/SELECTED_LAYOUT_REQUEST_PAIRS.html) · [前轮组件估算](official-long-input-experiment/SELECTED_LAYOUT_TRIAL.html)

[工作状态与下一步](WORK_STATUS.md) 记录最近一次能确认的板端状态、各阶段结果、复现入口及已知精度限制。

[多长度推理 HTML 报告](mlp-length-experiment/report.html#bf16-pipeline) 保留六轮实验。最新615 token暖态方法：上一合格候选P50 1743.55ms → 1579.62ms，候选P95 1642.71ms，P50下降9.40%；1秒目标未达成。34例原current数值门槛通过，confidence独立报告；不代表业务准确率。未部署；研究成果随本次main提交归档，适用范围和原始记录见第六阶段及 [WORK_STATUS.md](WORK_STATUS.md)。

这些多长度改动均为独立实验。下述 graph-half v1 记录的是 2026-10-18 的 d3-edge 服务发布。模型、编译图和 ARM 库按各实验目录的 `REPRODUCTION.md` 单独准备。

## 全过程版本与尝试记录

[从最初CPU部署到最终graph-half的统一记录](d3-edge-development-history.html)：按27个阶段记录每版的目标、实现、速度、输出差异、采用/淘汰原因，附所有留存配置结果、失败日志和源码快照索引。包含计时口径区分、469ms残差、失败注意力图、A72初筛纠正与confidence勘误。

文档由 `build_development_history.py` 读取本仓库历史证据生成；`python3 build_development_history.py --check` 可检查是否与留存资料一致，不运行模型或更改服务。

## 最新 graph-half 发布（2026-10-18）

318.51ms 暖态方法调用候选已接入 HTTP 服务，替换原三线程版本；旧容器保留用于回退。**快路径只适用于 batch1、115 token**，其他长度/形状回退CPU融合路径，不能承诺任意请求318ms。

- [最新 HTML 报告](cpu-tail-experiment/report.html)：实际LAN HTTP100次、19场景、图命中/回退、confidence勘误、发布与回滚步骤。
- [部署操作与适用范围](cpu-tail-deployment/README.md)：先8901验收、切换8900、失败恢复、实际回滚演练和调用示例。
- [318ms实验复现](cpu-tail-experiment/REPRODUCTION.txt)：冻结环境、源码构建、12个MLP图转换和性能对照。

下方独立HTML安装手册仍是基础安装入口；最新 graph-half 升级还需要仓库源码、历史工作区及另行生成的ARM库/12个RKNN图，不能只转交基础HTML就宣称包含318ms版本。二进制和模型不纳入Git。

## 给接收方的安装文档

下载并在浏览器打开 [从零安装与使用手册](d3-edge-rk3576-install-guide.html)。**只转交这一个 HTML 文件即可**：它内置全部适配源码、固定测试样本及源码初始化脚本，包含系统准备、依赖与模型下载、镜像构建、启动、调用、复现测试和排障步骤。

文档面向与测试机相同的 RK3576、16 GB 内存、ARM64 Armbian / Debian 13。它不依赖原测试服务器的 IP、模型缓存或现成镜像。接收方需自行安装并完成验收。

已有适配核心代码通过了原板端测试；独立从零安装路径尚未完整重跑，后续测试构建按用户要求停止。文档通过了命令语法、内嵌源码还原及桌面 / 手机页面检查。

## 验证结论

采用 **CPU + RKNN MatMul**：将 78 个文本线性层的矩阵乘交给 NPU，保留原始非因果注意力、GatedDeltaNet 因果扫描、视觉塔与 readout。保留官方原始 12 层结构及权重，没有额外裁剪。

| 2026-10-13 测试 | 结果 |
| --- | --- |
| 文本模型内成对基准 | 约 2.19–2.74 倍提速 |
| 小图片模型内基准 | 约 1.43 倍提速 |
| 单问题 HTTP | 6.686 → 2.822 秒，约 2.37 倍 |
| 优化 CPU → RKNN 输出差异 | 5 个用例、12 个问题选项一致；概率 / score 最大绝对差 0.00283136，容差 0.01 |

不同计时口径分别比较。概率差不等于任务准确率下降；结果只覆盖已测样本。超过 512 矩阵行回退 CPU，长文本吞吐和视频性能未验证。

官方 RKLLM 1.3.1 的本次适配能运行 NPU，但没有正确复现非因果注意力，因此不能作为已验证部署方案。`rkllm-d3-support/` 中的导出与策略组件没有接入官方 Runtime，不构成原生修复。

完整方法、原始数据与限制见 [HTML 测试报告](d3-edge-rk3576-report.html)。GitHub 文件页不执行 HTML；下载到本地后用浏览器打开。

## 目录

- `d3-edge-rk3576-install-guide.html`：可独立转交的安装手册，包含源码附录。
- `d3-edge-rk3576-report.html`：历史测试结果及运行状态核验快照。
- `build_install_guide.py` / `build_html_report.py`：两份 HTML 的生成器。
- `rknn-matmul-experiment/`：已测 RKNN 适配、微基准、边界及完整模型测试。
- `rkllm-experiment/` / `rkllm-d3-support/`：RKLLM 转换、诊断与未集成组件。
- `d3-edge-rknn-handoff/`：编写手册期间的独立安装源码草案；`manage.py` 仅通过有限单元检查，不代表安装流程验证完成。请以独立 HTML 手册为交付入口。
- 根目录与各实验的 JSON / 日志：保留的测试证据；IP、路径和服务状态均为当时环境快照。

## 固定版本与下载内容

- 模型：[`vllm-sr/d3-edge`](https://huggingface.co/vllm-sr/d3-edge/tree/20c745653400d191ecacd99c4eafc0c5900dc15e)，revision `20c745653400d191ecacd99c4eafc0c5900dc15e`。
- RKNN：[`airockchip/rknn-toolkit2`](https://github.com/airockchip/rknn-toolkit2/tree/2c723177a94fac65cbc036d722907ddbc91980cf)，Runtime 2.5.0；原板端驱动 0.9.8。
- Python 环境：Torch 2.12.0+cpu、Torchvision 0.27.0+cpu、Transformers 5.17.0；其余固定版本见 `d3-edge-rknn-handoff/requirements.lock`。

模型权重、RKLLM 转换产物、下载的 SDK 二进制 / 手册、浏览器缓存和运行数据不提交到 Git。安装文档中的工具从固定官方来源下载所需文件，并执行相应哈希检查。模型、基础镜像和 SDK 遵循各自许可；SDK 下载工具保留官方 LICENSE。

旧实验脚本含原测试机的固定目录，需先准备其依赖和模型才能运行；克隆本仓库不代表这些历史目录已经存在。

## 本地文档检查

在仓库根目录执行以下命令可校验文章、短文本结果和 nano 的七份概率对照，无需模型或 RK3576。概率对照保留个别选项超过 1 个百分点的原始结果。

```bash
python3 -B build_optimization_article.py --check
python3 -B d3.1-nano-npu-test/verify.py
python3 -B d3-engine/verify_results.py
python3 -B d3-engine/test_compare.py
```

基础安装手册、历史报告与交付工具另有以下检查。

```bash
python3 build_install_guide.py
python3 build_html_report.py
python3 -m unittest discover -s d3-edge-rknn-handoff -p 'test_manage.py' -v
bash -n rknn-matmul-experiment/reproduce-rknn.sh
```

以上不执行板端部署或完整性能测试。真实 NPU 验证和基准按安装手册在接收方自己的机器执行。
