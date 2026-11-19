# RK3576 上的 D3 多问题文本引擎

这里在 RK3576 上运行 d3.1-nano，按一整次 `system_one(state=..., questions=...)` 调用计时，包含分词、主干计算和决策头。对照保留输入全文、问题、选项、标签和工具定义。不同来源使用各自的验收方法。

| 来源 | 板上选取的输入 | 衡量什么 |
| --- | --- | --- |
| Jev 开发商 TypeSafe 的 WorkflowEvals | 按官方 `--limit 1` 规则，各取客服和发票工作流中 ID 排序第一的案例。执行官方分支与全部三套策略 | 各步骤的原始概率、整次模型调用耗时、最终动作集合与官方共识是否一致。共识由模型生成 |
| vLLM Semantic Router 的 Router Signal Suite | 中文 ACEBench 工具判断与 CrossWOZ 客服反馈，各 4 条 fresh 留出数据。每个实际出现的标签取按官方 token 数排序的 P50 和 P90，并以 ID 打破并列 | 官方 Noul 概率或五选一 Choice 分布、按标签的命中数以及板上耗时。8 条为固定抽样 |
| vLLM 发布的 d3.1-nano Decision Index 结果 | 从社区维护的 Decision Index 0.3 取订单、家电、合同三个完整请求，共 39 个 Choice 字段。按请求 SHA-256 对应发布方的同一条 MI300X 结果 | 日常服务按原选项顺序对照板上原版 CPU。复现发布方口径时另跑正反两序平均。发布方在 140,178 条可计分请求上的公开指数为 62.96 |

三个来源都固定到公开版本，复测脚本逐项检查输入或文件哈希。[TypeSafe 官方工作流](https://github.com/typesafe-ai/WorkflowEvals/tree/0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb) · [vLLM 官方留出集](https://huggingface.co/datasets/vllm-sr/router-signal-suite) · [vLLM 发布的 nano 逐请求结果](https://huggingface.co/datasets/vllm-sr/d3.1-nano-decision-index) · [Decision Index 0.3 复现工具](https://github.com/apolinario/decision-index/tree/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec)

13 条固定请求中，12 条完成同板原版纯 CPU 对照。按原选项顺序，四题一组的 FP32 注意力引擎在这 12 条请求上取得 36.3–79.1 倍的热态加速，58/58 个字段保持相同判断。172 个选项中有 170 个和 CPU 的概率差不超过 1 个百分点，两个超限选项属于同一道家电题。第 13 条是 55 题的完整发票，热态中位数仍需 20 分 30 秒。两个 TypeSafe 案例的最终动作均为 0/3 命中官方共识。[Decision Index 逐项对照](results/cpu-comparison.json) · [中文路由逐项对照](results/router-comparison.json) · [客服逐项对照](results/workflow-cpu-comparison.json)

可选的 `--checkpoint-math` 保留原版卷积、递推状态和 BF16 注意力，投影与 MLP 继续用 NPU。用最终引擎把同样的 12 条各跑两次，**58/58 个字段与 CPU 判断一致，172/172 个选项均通过 1 个百分点门槛**。最大选项概率差为 0.7527 个百分点，最大期望分差为 0.00283 分，相对原版纯 CPU 的热态加速为 13.0–33.9 倍。客服三套策略的动作集合也逐一与 CPU 相同。[该模式的完整运行](results/checkpoint-math.json) · [全部 172 个选项的 CPU、板端概率和差值](results/checkpoint-math-comparison.json)

| 执行方式 | 订单 4 题 | 家电 18 题 | 合同 17 题 | TypeSafe 客服 11 题 | vLLM 中文 8 条各自的耗时范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原版纯 CPU，每条一次 | 1,024.13 s | 3,590.04 s | 3,386.38 s | 2,495.35 s | 197.23–305.74 s |
| 默认 NPU + ARM，热态 | 20.36 s | 52.63 s | 74.83 s | 31.54 s | 5.20–7.08 s |
| NPU + `--checkpoint-math`，热态 | 57.97 s | 154.95 s | 243.10 s | 73.64 s | 14.25–18.94 s |

Router Signal Suite 适配器把原文放在 `state.request`，工具定义按原顺序放在 `state.tools`，题干和选项取自数据集作者的 `builder/tasks.py`。预期标签仅供评分。ACEBench 中文留出文件共有 365 行，其中 223 行需要工具。CrossWOZ 留出文件共有 1,995 行，其中 `SAT` 有 146 行，`NO_FEEDBACK` 有 1,849 行，模型回答时始终面对官方五个选项。本次按标签等量抽样，表中的命中数适用于选出的 8 条。[样本提取与 D3 输入构造](prepare_router_suite.py)

## TypeSafe 官方工作流

首个客服案例是旧手机号码导致用户收不到登录验证码。模型在 11 个问题中与官方共识一致 9 个，给“一般咨询”69.42%、“账户访问”28.99%。官方策略据此返回一般政策说明，三个策略要求的账户恢复动作均未命中。热态模型调用为 31.54 秒，包含 1,481 个输入 token，进程峰值 RSS 为 6.36 GiB。此类错误需要以最终动作集合验收。

同板原版 CPU 用 2,495.35 秒完成该客服请求，进程峰值 RSS 为 3.90 GiB，新引擎热态加速 79.1 倍。11 个字段的最终判断全部一致，36 个选项的最大概率差为 0.4639 个百分点，两个 Score 问题的最大期望分差为 0.00341 分。把 CPU 输出送入官方工作流后，三个动作集合逐一与新引擎相同。下文的完整发票只测了新引擎与官方共识，未测板上纯 CPU 耗时和概率。[客服纯 CPU 记录](results/workflow-cpu.json) · [客服逐选项对照](results/workflow-cpu-comparison.json) · [纯 CPU 策略重放](results/workflow-cpu-policies.json)

首张发票 `ap_00000` 有 55 个问题，状态段为 8,902 token，最长问题连同状态有 10,605 token。若逐题重复原文，所有输入长度相加为 515,702 token。共享状态后仍需处理 34,994 token，其中 26,092 来自问题本身。模型在 45/55 个问题上与官方共识一致。三个策略均把“疑似重复”列为首要动作，官方共识要求争议款项复核。两次热调用分别为 1,225.57 和 1,234.16 秒，中位数为 **20 分 30 秒**，两次返回的所有概率完全相同。首轮有 644.85 秒用于 CPU 注意力，259.75 秒用于 NPU 矩阵乘与搬运。进程峰值 RSS 为 8.53 GiB，容器 cgroup 记录的峰值为 5.80 GiB，交换空间用量为零。文件映射页与驱动缓冲区的内存计量口径不同。客服与发票各用一次模型调用，官方代码在宿主机计算三个策略的最终动作。以上耗时只包含 RK3576 模型调用。工作流的最终动作、共识动作及逐策略评分均有记录。[板端逐次响应](results/workflow-npu.json) · [官方策略评分](results/workflow-policies.json) · [客服测试数据](https://huggingface.co/datasets/typesafeAI/evalsafe-customer-service) · [发票测试数据](https://huggingface.co/datasets/typesafeAI/evalsafe-invoice-processing)

发票问题 `prior_1` 要判断先前账单和当前账单的关系。板端给“同一付款义务”75.92%，给“较早账期或项目阶段”18.99%。官方共识给这两个选项的概率分别为 1.05% 和 97.90%。该字段会进入官方的重复付款检查。[官方标签数据](https://huggingface.co/datasets/typesafeAI/evalsafe-invoice-processing/blob/6beeb2d2acd65c086c835022f5f4d7434114cafc/data/questions.parquet) · [工作流付款事实提取](https://github.com/typesafe-ai/WorkflowEvals/blob/0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb/evals/invoice_processing/structured/facts.py)

34,994 是有效输入 token 数。按实测分词长度和原检查点的分组函数核算，发票的问题段在原版分组中需要填充到 34,975 个位置。新引擎将其拆成实际执行的 15 组，含填充共 27,771 个位置，另加一次 8,902 个位置的状态段。客服问题段的含填充位置从 2,750 个降到 1,558 个。[检查点的 `state_once_batches`](https://huggingface.co/vllm-sr/d3.1-nano/blob/71d06677f56e243042fe5d5f983804648a5689e8/d3_runtime.py)

## vLLM 中文路由留出集

| 官方任务 | 样本数 | 实际 D3 输入 token | 纯 CPU 各请求 | 新引擎各请求热态中位数 | 与 CPU 最大概率差 | 两种后端命中标签 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ACEBench 中文，当前应否调用给定工具 | 4 | 253–347 | 225.76–305.74 秒 | 5.52–7.08 秒 | 0.4396 个百分点 | 2/4 |
| CrossWOZ 中文，判别用户反馈 | 4 | 218–241 | 197.23–217.29 秒 | 5.20–5.77 秒 | 0.3699 个百分点 | 3/4 |

新引擎对每条请求完整运行三次。首条剔除包含权重打包的首次调用。此表按 `noul ≥ 0.5` 判断是否调用工具，按最高概率判断反馈类型。vLLM 的完整榜单另有使用开发集校准阈值的指标。模型对四条工具请求均选“是”，其中官方标签为“否”的两条分别给出了 82.92% 和 94.12% 的调用概率。反馈案例 `001099` 从上句转为餐馆推荐，模型选满意反馈（57.73%），官方标签是无反馈。八条的进程峰值 RSS 为 6.34 GiB。[板端分布和逐次耗时](results/router-npu.json) · [vLLM 官方评分脚本](https://huggingface.co/datasets/vllm-sr/router-signal-suite/blob/fa08b2a642df30955ad2ad2206d74050c7f12b5c/scoring/score.py)

原版 CPU 对每条请求实跑一次，峰值 RSS 为 3.38 GiB。逐条相除，工具判断加速 40.9–47.7 倍，反馈分类加速 36.3–38.7 倍。8 个判断全部一致，28 个选项均通过 1 个百分点的概率门槛。[纯 CPU 原始运行](results/router-cpu.json) · [28 个选项的概率对照](results/router-comparison.json)

## Decision Index 整单耗时与业务标签

2026-11-19，16 GiB NanoPi R76S，d3.1-nano。两列引擎耗时使用相同的 NPU 矩阵乘内核、FP32 状态扫描以及 4 题分组。FP32 列采用新的 CPU 缩放点积注意力（SDPA）与卷积缓存。NPU 输入和权重为 FP16，累加为 FP32，输出回到 BF16。原始决策头使用 FP32。

| 完整文本请求 | 状态 token | 每题最长上下文 | 逐题提示 token 之和 | 状态共享后实际输入 token | CPU 注意力 BF16 | CPU 注意力 FP32 | FP32 方案对业务标签 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 客服订单，对 4 个工具分别判断是否调用 | 784 | 885 | 3,538 | 1,186 | 36.04 s | 20.36 s | 4/4 |
| 家电模拟器，一次判断 18 个控制字段 | 1,016 | 1,193 | 20,051 | 2,779 | 99.76 s | 52.63 s | 18/18 |
| 保密协议，对 17 个条款假设分类 | 1,939 | 2,054 | 34,707 | 3,683 | 178.89 s | 74.83 s | 15/17 |

表内均为同一进程串行运行三条输入、每条三次的热态中位数，首条订单剔除首次调用。首次订单含权重打包分别为 68.01 和 52.06 秒，最终 FP32 引擎加载模型另用 13.61 秒。FP32 进程峰值 RSS 从订单的 6.38 GiB 增至合同后的 6.63 GiB，驱动另会分配 DMA 缓冲区。卷积层只需保留最后三个 token 的历史。原缓存持有指向整个投影的视图。按层复制这三项后，合同请求少持有 408 MiB，进程峰值从初次 FP32 试验的 7.02 GiB 降至 6.63 GiB。两轮在所有 39 个字段上的返回值完全相同。[最终打包运行结果](results/rknn-fp32.json)

同一块板上的原版纯 CPU 使用同一检查点，保留其 BF16 推理和状态复用。表中 CPU 每单实跑一次，引擎使用上表的热态中位数。两列均不计模型加载。内存列分别给出两种运行的进程峰值 RSS，单位为 GiB。

| 完整请求 | 纯 CPU | 新引擎 | 热态加速 | CPU / 引擎峰值 RSS | 与 CPU 选择一致 | 所有选项最大概率差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 订单 | 1,024.13 s | 20.36 s | 50.3× | 3.55 / 6.38 | 4/4 | 0.0086 个百分点 |
| 家电 | 3,590.04 s | 52.63 s | 68.2× | 4.41 / 6.51 | 18/18 | 1.5932 个百分点 |
| 合同 | 3,386.38 s | 74.83 s | 45.3× | 4.45 / 6.63 | 17/17 | 0.2993 个百分点 |

三个请求共 108 个选项。106 个和 CPU 相差不超过 1 个百分点。按请求验收，订单和合同通过，家电越过概率门槛。[纯 CPU 逐题响应](results/cpu.json) · [CPU 与新引擎的 108 项概率对照](results/cpu-comparison.json)

订单热调用的 BF16 路径有约 20 秒落在 NPU 矩阵乘、卷积和状态扫描以外。保留真实 Q/K/V 和掩码重放 12 次注意力，BF16 合计 15.51 秒，FP32 合计 2.67 秒。同一次测量中 36 次小投影合计 0.89 秒。把 FP32 注意力接回完整模型后，表中三条请求均变快。两种精度仍分别接受原版 CPU 的概率核对。[注意力测量与计算](cpu_attention.py)

逐项对照文件沿用数据集的字段和选项 ID。可用附带的题干与选项对照表查询每个 ID 的业务含义。订单对话询问已下单运动鞋的配送状态。四题分别输出“应该调用该工具”的概率，每题的另一个选项为“否”。[39 个字段的题干和原始选项](results/decision-fields.json)

| 工具 | 官方纯 CPU | 新引擎，FP32 注意力 | 变化（百分点） |
| --- | ---: | ---: | ---: |
| 库存管理 | 0.43827% | 0.42966% | −0.00861 |
| 商品搜索 | 0.17753% | 0.17250% | −0.00503 |
| 查询订单状态 | 99.76710% | 99.76535% | −0.00175 |
| 获取商品详情 | 0.27756% | 0.27032% | −0.00723 |

家电请求的 18 个字段共有 49 个选项。只有客厅灯亮度这一个字段出现超过 1 个百分点的差异。亮度 35 从 CPU 的 36.56% 升到 38.15%（+1.5932 个百分点），亮度 55 从 63.15% 降到 61.56%（−1.5919 个百分点）。两种后端最终都选中正确的 55。对使用概率阈值的下游，这一差异仍计为未通过。

合同请求还暴露了模型回答错误时的概率。FP32 注意力方案将 `nda-16` 和 `nda-19` 判为 `Entailment`，概率分别为 93.88% 和 90.76%。原版纯 CPU 也选中 `Entailment`，分别给出 93.61% 和 90.46%。数据集标签分别是 `NotMentioned` 和 `Contradiction`。

按 vLLM 发布方的双选项顺序重跑，RK3576 与公布的 MI300X 输出在全部 39 个字段上选项相同，合同也错在同两题。订单、家电、合同相对于 GPU 的最大选项概率差依次为 **0.0102、0.3456、0.4130 个百分点**。双序热态分别为 29.54、89.58、115.14 秒，输入 token 分别为 1,588、4,542、5,427。这些时间包括将两份选项分布按标签对齐再平均。与板上单序 CPU 的偏差另计。[双序板测](results/jev-double.json) · [MI300X 与板端的每个选项](results/gpu-comparison.json)

## 家电用例的概率差来自哪些算子

固定同一个检查点、同一份 18 题请求、FP32 决策头和 NPU 矩阵乘，分别替换分组、状态计算和注意力。五组实验都选对 18/18 个字段。表中最大差异按每一个选项与同板原版 CPU 核对。

| 问题分组上限 | 卷积与递推状态 | CPU 注意力 | 整单热态 | 相对纯 CPU | 最大概率差 |
| ---: | --- | --- | ---: | ---: | ---: |
| 4 | ARM 原生内核 | FP32 | 52.63 s | 68.2× | 1.5932 个百分点 |
| 64 | ARM 原生内核 | FP32 | 70.90 s | 50.6× | 1.5932 个百分点 |
| 4 | ARM 原生内核 | 检查点 BF16 | 99.76 s | 36.0× | 1.4456 个百分点 |
| 4 | 检查点原版 | FP32 | 107.83 s | 33.3× | 1.0094 个百分点 |
| 4 | 检查点原版 | 检查点 BF16 | 157.59 s | 22.8× | **0.7527 个百分点** |

将 18 题合成一组，使进程峰值 RSS 从 6.51 升到 7.19 GiB，最大偏差落在同一个选项上且数值相同。组合保留两类检查点算子后，49 个选项全部通过 1 个百分点门槛。这个模式继续将投影和 MLP 放在 NPU 上。[五组运行的逐选项比较与原始结果哈希](results/home-ablations.json)

家电单独复测的 157.59 秒与最终 12 条同进程测量的 154.95 秒来自两次独立运行，两个进程对该请求返回的全部概率相同。跨任务复核原版 CPU 对照时可运行下列命令。

```sh
cd d3-engine/results
python3 ../compare.py cpu.json checkpoint-math.json --also-cpu router-cpu.json --also-cpu workflow-cpu.json
```

## 下一轮的性能依据

针对 55 题发票，推荐先验证分块长注意力和 K/V 复用，再决定 Mali GPU 的适用长度。候选的硬件分工、内存预算和官方用例验收线见 [RK3576 推理优化方案](RK3576-OPTIMIZATION-PLAN.md)。

- **长发票先看注意力**。55 题首次实测有 644.85 秒花在注意力，占 52.6%。其中状态的自注意力是方阵，问题读取状态和本题时是长方阵。已有 edge 实验内核的全键路径要求方阵，最大长度为 8,192。这张发票的状态段本身就有 8,902 token。下一版可以让 NPU 计算分块的 QK 与 PV，保留 FP32 的跨键块最大值、归一化分母和累加结果，并复用同一状态的 K/V。验收必须包括真值为假的屏蔽位置、最后一个不满块和不同题长。即使解决这一部分，本次调用的其余环节仍占约 581 秒。
- **常驻服务**。在各样本最后一次 FP32 热调用中，NPU 矩阵乘与布局搬运合计占 9.56、21.94、30.03 秒，FP32 自注意力分别占 2.07、7.02、15.66 秒。可以针对同一份输入的多路投影复用转换结果，并在内存允许时实测更大的行块。对长状态，注意力可以按查询块调度，每个查询仍需访问完整的有效键。每个候选都应回到相同整单测耗时和概率。
- **冷启动**。首次订单用时比热调用多约 32 秒，首次调用会创建 RKNN 上下文并打包权重。可单独测量预打包权重的加载耗时、常驻内存和文件描述符数量。复用的打包文件需能区分权重哈希、SDK 版本和矩阵布局。
- **业务验收**。这三条公开请求覆盖了多工具、多字段和长文本，输出均为 Choice。面向线上容量时，应回放业务文本和问题数的实际分布，在目标到达速率下计入排队、超时与冷启动，报告整单 P50/P95。对于有动作阈值的字段，还需检查概率变化会翻转多少个业务动作。

## 执行方式

| 检查点 | 同一个请求内能复用什么 | 当前执行方式 |
| --- | --- | --- |
| d3.1-nano，`prefix_bidir` | 文本状态的表示与具体问题无关。双向注意力层保留状态段的 K/V，线性注意力层保留 FP32 递推状态和卷积尾部 | 状态算一次。问题分组，默认每组最多 4 个，分别读取同一状态，用各自的长度选取最后一个有效 token |
| d3-edge，`noncausal_full_attention` | 每题可见完整的状态和该题，题目会影响状态的隐藏表示 | 沿用检查点自己的整行前向与掩码，各题分别计算。逐 token 的大矩阵乘可使用同一套 NPU 内核 |

检查点名里的参数量不足以选择执行方式。例如 d3-nano 和 d3.1-nano 同为 2B 级，前者的全注意力层采用整行双向掩码，后者支持分开的状态段和问题段。引擎读取 `backbone_type`、`attention_mode` 和决策头精度。此版本的入口接收文本，其他权重规模、视觉输入、量化模型和多实例容量需要独立验证。[d3-nano 配置](https://huggingface.co/vllm-sr/d3-nano/blob/8198c9f763b046605b2f4827b4999971fa2c92c4/decision_config.json) · [d3.1-nano 检查点](https://huggingface.co/vllm-sr/d3.1-nano/tree/71d06677f56e243042fe5d5f983804648a5689e8)

这轮在同一块板上把同一条订单原文和 4 个问题分别交给两个检查点。新引擎用 0.6B d3-edge 实跑 85.33 和 87.01 秒，热态中位数 86.17 秒，峰值 RSS 1.83 GiB。用 2B 级 d3.1-nano 为 20.36 秒和 6.38 GiB，整单快 4.23 倍。两边均答对 4/4。edge 的整行双向模式处理 3,538 个有效输入 token，nano 的前缀双向模式处理 1,186 个。两条运行得到的原文及选项哈希相同。edge 的短输入仅用于预热。[本次 edge 原始运行](results/edge.json) · [本次 nano 原始运行](results/rknn-fp32.json)

官方 nano CPU 对照也启用状态复用。CPU 和 RKNN 比较使用相同的输入 token 计量。仓库此前的 d3-edge r078 在同一条 18 字段家电请求上实际计算了 20,051 个输入 token，耗时 270.82 秒，字段命中 11/18。这次 2B 级 d3.1-nano 用 52.63 秒得到 18/18，整单快 5.15 倍。这个历史对比同时改变了模型、注意力规则和引擎版本。新引擎本身的加速比由同一 nano 检查点对照纯 CPU 得出。[历史样本匹配和结果哈希](results/board.json)

家电样本的 18 个问题块实际共 1,763 个 token，单块长度从 83 到 177。原版将这 18 行一起补到 177，逐层需要处理 3,186 个问题位置。沿用原版的长度排序，每 4 题各自补齐后只需处理 1,822 个位置，少算 1,364 个填充位置。输入计费、题目顺序和选项数保持原有定义。

1. 旧适配只接受单个序列且最多 128 行。新适配把**逐 token 独立**的注意力投影和 MLP 按 128 行切块，算完恢复原始批次和序列形状。一个块可以包含两个不同问题的 token，因为这些算子逐行独立。注意力、位置编码、归一化和选项读出仍由原检查点调度。数值超出原生内核范围时整层退回 CPU。[NPU 调度与模型入口](engine.py)
2. 对 d3.1-nano，ARM 卷积接受前一段留下的卷积历史。FP32 分块扫描能返回状态，也能从同一份只读状态开始多个问题分支。每个分支只看本题，组内只补齐到最长问题。状态只在本次调用内存活。不同请求通过同一把锁访问可复用的 RKNN 缓冲区。[ARM 扫描](state_scan.c) · [Python 绑定](state_kernels.py)
3. 决策头沿用检查点原始 FP32 权重、温度、选项顺序和概率计算。用同一权重的官方 CPU 作数值参照，用数据集标签衡量模型答对多少字段。两者分别记录。[逐选项比较](compare.py)

板端工作进程可以通过原来的请求结构调用引擎。在下面复测容器的 Python 路径和动态库环境下，模型加载一次即可处理多条独立请求。常驻入口和 `D3Engine` 的 `attention="auto"` 默认对 `prefix_bidir` 采用 FP32 注意力，对整行双向检查点采用原版注意力。基准工具的 `--attention` 单独显式指定，便于复现每一种对照。

```python
import os

from d3_runtime import D3
from engine import D3Engine

model = D3.from_pretrained(os.environ["D3_CHECKPOINT"], device="cpu", batch_size=1, verify="fast")
with D3Engine(model) as engine:
    result = engine.system_one(state=order_state, questions=tool_questions)
```

同一容器也可以从标准输入接收 JSON Lines。每行包含 `state` 和 `questions`，输出一行含 `answers` 与 `usage` 的官方响应。进程启动时加载模型，后续请求复用同一实例。下面的命令把多个公开用例的原始请求直接送入引擎。[常驻推理入口](infer.py)

```sh
jq -c '{state, questions}' "$lab/src/order.json" "$lab/src/home.json" |
  bash "$lab/src/run-board.sh" "$lab" "$checkpoint" "$run_id-infer" infer
```

要在 nano 上保留检查点原版的状态算子和注意力，在常驻命令末尾加 `--checkpoint-math`。Python 入口等价于 `D3Engine(model, attention="checkpoint", checkpoint_state_kernels=True)`。使用基准入口复现该模式时，传 `--checkpoint-state-kernels --attention checkpoint`。

## RKLLM 的适配边界

RKLLM 1.3.1 提供 token 输入和最后一层隐藏向量输出，D3 可以在外部接 FP32 决策头。公开 C 接口中的 `encoder_mask` 用于交叉注意力。实现这两个检查点还需要控制文本**自注意力**的可见范围、每层的真实类型，以及线性注意力的递推状态。仓库此前在同一版本上给 d3-edge 导出 `causal=false` 并实际运行，4 个 choice 只对齐 2 个，最大选项概率差 75.27 个百分点。该结果来自 edge。这里用可控制每个算子的 RKNN MatMul 实现 nano 的 NPU 路径。[官方 RKLLM 接口](https://github.com/airockchip/rknn-llm/blob/f7390530443bf84f0394255a449d7cbe81e69d1c/rkllm-runtime/Linux/librkllm_api/include/rkllm.h) · [edge 板测原始记录](../rkllm-experiment/experiment-result.json)

接入新的 RKLLM Runtime 时，先验证相同输入下的注意力语义。在整行双向模式，改变后面的 token 应能影响前面位置的隐藏向量。在前缀双向模式，替换某个问题应保留状态段隐藏向量，也应保留另一个问题的输出。探针通过后再对完整业务请求的所有选项概率和延迟作比较。[已有 Runtime 接口适配要求](../rkllm-d3-support/native-integration-requirements.json)

## 板端复测

在宿主机从冻结的 Decision Index 0.3 样本提取输入。首次复测可按维护者的 `suite rebuild` 和 `suite import` 步骤重建数据，把第一行命令的输入换成 `suite-0.3/selected-rows.jsonl.gz`。这两个来源提取出的三条请求具有相同哈希。`--gpu-reference` 流式读取发布方的完整结果，校验文件哈希，再保留三个相同请求。中文留出集的准备脚本会下载固定版本的两个公开文本文件和任务定义，校验完整文件哈希。原始输入保存在本地工作目录。[Decision Index 复建说明](https://github.com/apolinario/decision-index/blob/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec/README.md)

```sh
python3 d3-engine/prepare_cases.py output/decision-index-audit/sample-1000.jsonl.gz output/d3-engine \
  --gpu-reference output/d3-engine/published-jev.json
python3 d3-engine/prepare_router_suite.py output/d3-engine/vllm output/d3-engine/router
```

在板上用仓库的源码构建两个原生库。设置 `lab` 为独立工作目录，`checkpoint` 为固定检查点，`D3_SDK` 为含 RKNN 头文件和 `librknnrt.so` 的目录，`run_id` 为本次容器名前缀。将上一步生成的三个 JSON 放入 `$lab/src`。如果已有相同编译产物，也可以用 `D3_NATIVE` 指向其目录。[板端构建](stage-board.sh) · [RKNN 原生内核构建](../d3.1-nano-npu-test/build.sh)

```sh
export D3_SDK
bash d3-engine/stage-board.sh "$lab" "$D3_SDK"
cp output/d3-engine/{order,home,contract}.json "$lab/src/"
bash "$lab/src/run-board.sh" "$lab" "$checkpoint" "$run_id-cpu" cpu \
  /out/cpu.json /bench/order.json /bench/home.json /bench/contract.json
bash "$lab/src/run-board.sh" "$lab" "$checkpoint" "$run_id-rknn" rknn \
  /out/rknn.json /bench/order.json /bench/home.json /bench/contract.json --attention fp32 --repeats 3
python3 "$lab/src/compare.py" "$lab/results/cpu.json" "$lab/results/rknn.json"
```

复测中文路由样本时，将准备好的 `tool_need-*.json` 和 `feedback-*.json` 放到板上的 `$lab/src`，分别在相同容器中执行。以下命令从文件名生成容器内的路径，保持两种后端的输入顺序相同。

```sh
router=()
for file in "$lab"/src/tool_need-*.json "$lab"/src/feedback-*.json; do
    router+=("/bench/$(basename "$file")")
done
bash "$lab/src/run-board.sh" "$lab" "$checkpoint" "$run_id-router-cpu" cpu /out/router-cpu.json "${router[@]}"
bash "$lab/src/run-board.sh" "$lab" "$checkpoint" "$run_id-router-npu" rknn \
    /out/router-npu.json "${router[@]}" --attention fp32 --repeats 3
python3 "$lab/src/compare.py" "$lab/results/router-cpu.json" "$lab/results/router-npu.json"
```

运行器限定 CPU 4–7、Torch 3 个算子内线程、单个模型进程、容器内存 8 GiB、无容器网络，NPU 模式额外挂载 `/dev/dri` 并设置 `nofile=4096`。首轮沿用容器默认的 1,024 个文件描述符时，RKNN 在创建 DMA 缓冲区时出现 `EMFILE` 并退出。原始纯 CPU 模式只加载官方检查点。数值比较要求每题选择一致、每个选项概率差不超过 0.01，另外报告置信度差值。进程峰值 RSS 包含映射权重，在同一进程内累计。首条请求包含 NPU 权重打包，后面每条请求仍重新计算自己的状态。[测量入口](bench.py) · [容器参数](run-board.sh)

对照 vLLM 公布的 MI300X 结果时，在 RKNN 命令增加 `--permutation-average`，输出到独立的 `jev-double.json`。相同的选择题需获得两份分布，在原选项上逐项取均值。原版日常服务的默认值是单序。发布方用了较早的运行时，主权重、决策头和分词器与本次固定版本一致。比较脚本校验请求哈希、字段、完整选项和输入 token 计量。[发布方环境](https://huggingface.co/datasets/vllm-sr/d3.1-nano-decision-index/blob/767d71ddb868c0c2be015ea0c5b9110cbbd86451/runs/d3.1-nano/environment.json)

```sh
python3 d3-engine/compare.py --published-jev output/d3-engine/published-jev.json "$lab/results/jev-double.json"
```

### TypeSafe 工作流复测

工作流规则与评分器在宿主机运行，实际推理由板上离线容器执行。先在宿主机准备官方代码和所需的公开数据。首次运行会按真实策略为各案例生成第一个模型请求。

```sh
work=output/d3-engine/typesafe
git clone https://github.com/typesafe-ai/WorkflowEvals.git "$work/kit"
git -C "$work/kit" checkout 0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb
uv sync --directory "$work/kit" --frozen --python 3.13
hf download --repo-type dataset typesafeAI/evalsafe-customer-service --revision b1342f5a704587dbc465867c38c2694348ff86e4 \
  --include dataset.json data/cases.parquet data/questions.parquet --local-dir "$work/customer-service"
hf download --repo-type dataset typesafeAI/evalsafe-invoice-processing --revision 6beeb2d2acd65c086c835022f5f4d7434114cafc \
  --include dataset.json data/cases.parquet data/questions.parquet --local-dir "$work/invoice-processing"
PYTHONPATH="$work/kit" "$work/kit/.venv/bin/python" d3-engine/workflow_eval.py "$work" output/d3-engine/workflow-inputs
```

将生成的 JSON 复制到板上的 `$lab/src`，以相同的 RKNN 命令运行这些输入。把运行器产出的 JSON 拷回宿主机，作为 `workflow_eval.py` 的后续位置参数，并加上 `--report` 指定报告文件。若模型输出触发后续节点，工具会写出该节点的完整输入，重复执行直至获得三个策略的最终动作。仅复测客服时，给 `workflow_eval.py` 加 `--workflow customer-service`，再用 `compare.py --request` 和该节点的完整 ID 对照两份运行记录。这里的动作是评测结果，策略代码只生成动作记录。[工作流适配和评分](workflow_eval.py)

## 验证记录

- 在 RK3576 的固定镜像内运行 `python -u test_engine.py`，9 项通过。覆盖默认精度与检查点模式、问题分组与顺序、掩码、分块矩阵乘尾块与回退、卷积及递推续算、缓存占用和实例并发。
- 通过 `run-board.sh … infer` 和 `run-board.sh … infer --checkpoint-math` 分别启动两种模式的实际常驻进程，依次输入 vLLM 留出请求 `feedback-fresh-crosswoz-000261`、缺少 `state` 和 `questions` 的 JSON 对象、同一条留出请求。标准输出可逐行解析，中间返回 `invalid_request`，前后两次的答案和 token 计数均与各自模式的板测记录逐项相同。
- 宿主机的 `python3 -B d3-engine/test_compare.py` 四项通过。`shellcheck d3-engine/{run-board,stage-board,build-state}.sh` 和 `git diff --check` 通过。官方重建的 Decision Index 全集和本地冻结样本提取出的三个输入及其字段对照表逐字节相同。
- 从原始响应重新生成五份对照报告，与归档一致。默认模式单序 Decision Index 的比较器因家电两项偏差返回 1，其余四份报告返回 0。实测结束后独立容器均已退出并移除，原 `decision3-edge-graph` 容器处于 `exited`。[板端环境与核对记录](results/board.json)
