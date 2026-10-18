# 不改模型，重做推理执行：在 RK3576 上把 d3-edge 从秒级优化到 300 毫秒级

> 技术总结草稿，可继续编辑发布。依据截至 2026-10-18、提交 `3b930ed` 的代码与留存验收记录整理。部署状态指记录中的已验收状态，不是实时服务探测；本文整理没有重新运行板端性能测试。

在边缘设备上部署模型，推理速度往往取决于整条执行链路：算子分配给谁、张量如何转换、数据经过多少次复制，以及 CPU 和 NPU 如何协作。

这次我们在 RK3576 上优化 d3-edge，始终保持一个前提：**保留原始模型权重和 12 层网络结构，通过优化推理引擎降低延迟。**

没有重新训练或微调，没有蒸馏、裁层，也没有采用 INT8、INT4 权重量化。工作集中在执行层，包括 CPU/NPU 混合计算、原生算子、数据布局、缓冲区复用、子图编译和调度。

目前已经得到两组需要分别看待的结果：

| 状态 | 输入与测量范围 | P50 | P95 |
| --- | --- | ---: | ---: |
| **已部署：graph-half v1** | batch1、115 token，100 次 LAN HTTP | **325.27ms** | **328.92ms** |
| **最新实验候选，尚未部署** | batch1、615 token，90 次暖态模型方法调用 | **1579.62ms** | **1642.71ms** |

短输入版本的模型方法中位数为 318.51ms；325.27ms 是独立测得的真实 HTTP 结果。615 token 的最新结果不包含 HTTP、请求排队和冷启动。

因此，“300 毫秒级”有明确适用范围，不能直接推广到任意长度、图片输入或并发请求。

## 1. “不改模型”，具体保留了什么

这里所说的模型不变，指原始检查点权重、网络结构以及关键计算语义保持不变。

d3-edge 同时包含完整注意力和 GatedDeltaNet 线性注意力。优化过程中，完整注意力必须保留非因果语义，让 query 能够访问所有未被屏蔽的 key；GatedDeltaNet 必须保留因果递推及跨块状态。不能为了更快而截断输入、缩短 K/V，或把扫描分块当作互不相关的新序列。

执行引擎允许变化。例如，原始 BF16 权重可以生成 FP32 或 FP16 执行副本，多个算子可以融合，独立矩阵乘可以并行，固定形状可以编译成子图。

这也意味着，**模型不变不等于浮点结果逐位相同**。执行副本转换为 FP16 时存在范围与舍入问题；计算精度、累加顺序和算子实现变化也可能引入数值差异。因此，不能把“没有 INT8/INT4 量化”写成“所有计算精度都没变”，而应通过回归验证决定候选是否合格。

整个优化过程，实际上是在这组约束下重新安排同一个模型的计算。

## 2. 从整模型 RKLLM 尝试，转向可控的混合执行

最初，我们尝试过直接通过 RKLLM 运行整个文本编码器。这个方向很直接，但遇到了语义兼容问题。

模型实际的完整注意力层与线性注意力层排列，并不符合当时 Runtime 的层类型推断方式。进一步适配后，虽然能够运行 NPU，导出信息也写入了 `causal=false`，实际结果仍更接近因果注意力。

在诊断样本中，最大概率差达到约 0.753，4 个 choice 只匹配了 2 个。模型能加载、NPU 有负载、接口能返回，都不足以证明执行正确。

最终，我们没有采用这条整模型 RKLLM 路线，而是转向公开 RKNN MatMul API，建立可控的 CPU/NPU 混合执行引擎。

需要特别澄清的是，后文约 6.686 秒的 HTTP 基线来自早期优化 CPU 路径，不能将它写成“已验证 RKLLM 版本的延迟”。RKLLM 原型有独立的非 HTTP 计时，但因为语义验证失败，没有成为正式部署基线。

证据入口：[RKLLM 实验结果](rkllm-experiment/experiment-result.json)、[隐状态诊断](rkllm-experiment/logs/hidden-diagnosis.json)。

## 3. 第一阶段：优化 CPU 算子，再迁移大型矩阵乘

### 3.1 缓存 FP32 执行副本，保留 BF16 激活边界

早期 CPU 路径中，BF16 Linear 和 Conv 的表现不理想。我们缓存 FP32 权重与偏置副本，使用 FP32 执行这些算子，再将输出转换回原来的 BF16 激活边界。

这既避免了每次请求重新转换权重，也让主要计算进入更适合当前 CPU 环境的执行路径。原始模型文件保持不变，缓存副本只属于运行时。

下面是实际实现中的 Linear 路径：

```python
def linear_forward(self, inputs):
    return functional.linear(inputs.float(), self._cpu_fp32_weight,
                             self._cpu_fp32_bias).to(inputs.dtype)
```

代码引用：[cpu_optimized_server.py](cpu_optimized_server.py)，重点看 `optimize()`、`linear_forward()` 和 `conv_forward()`。`optimize()` 将执行副本注册为非持久 buffer，不将它们作为新模型参数写回检查点。

### 3.2 将 78 个文本大型线性层交给 NPU

随后，将 78 个符合条件的文本大型线性层迁移到 RKNN MatMul。注意力、GatedDeltaNet 扫描、视觉塔和 readout 等部分继续保留在原有或后续优化的 CPU 路径中。

这一阶段采用长期持有的矩阵句柄和预打包权重。模型加载、权重准备与句柄初始化放在暖态请求之外，请求直接复用这些资源。

早期适配器对超过 512 行的输入保留 CPU 回退。这个数值是当时软件的资格边界，不能当作硬件理论上限。

代码引用：[rknn_linear.py](rknn-matmul-experiment/rknn_linear.py) 的 `Engine.create()`、`Engine.run()` 和 `install()`；[matmul_bridge.cpp](rknn-matmul-experiment/matmul_bridge.cpp) 是对应的公开 SDK 桥接实现。

### 3.3 从秒级到亚秒级的 HTTP 演进

| 阶段 | 单问题 HTTP 耗时 | 测量说明 |
| --- | ---: | --- |
| 早期优化 CPU 路径 | 6.686s | 单次抽查 |
| 初版 CPU + RKNN | 2.822s | 3 次中位数 |
| 三线程优化版正式部署 | 859.21ms | 100 次中位数 |
| graph-half v1 正式部署 | 325.27ms | 100 次中位数 |

这些是不同阶段的历史测量，样本量和运行条件存在差异，适合展示演进过程，不能替代统一协议下的最终对照，也不应直接计算为严格端到端提速倍数。

证据入口：[初版 RKNN 结果](rknn-matmul-experiment/experiment-result.json)、[三线程部署结果](optimized-deployment/deployment-result.json)、[graph-half 部署结果](cpu-tail-deployment/deployment-result.json)。

## 4. 第二阶段：处理 CPU 热点与尾延迟

### 4.1 因果卷积、RMSNorm 和扫描冗余

大型矩阵乘上 NPU 后，下一轮瓶颈暴露在 CPU 上。一次总耗时约 2.267s 的剖析中，因果卷积就占了约 1.207s。此时继续只盯着矩阵乘，已经很难获得主要收益。

我们针对四抽头 depthwise 因果卷积实现原生融合路径，保留必要的 BF16 舍入；随后融合 RMSNorm，减少框架调度和临时张量。同时，针对零初态、无需返回最终状态的扫描调用，删除不影响结果的冗余工作。

文本完整注意力也切换到更合适的 FP32 SDPA 路径。这个优化没有直接推广到视觉部分：早期全局替换曾使图片样本超出数值门槛，最终将资格范围收缩到文本注意力。

**算子名称相同，不代表不同输入和模块可以共享同一条优化路径。**

代码引用：[cpu_attention_optimized.py](latency-experiment/cpu_attention_optimized.py)、[native_norm.py](latency-experiment/native_norm.py)、[原生算子实现](latency-experiment/native_norm.c)。验证范围见 [latency-experiment/README.md](latency-experiment/README.md)。

### 4.2 固定计算线程，减少系统活动干扰

在固定使用 CPU 4–7 的条件下，4 个计算线程能得到较好的部分中位数，却出现明显慢峰。最终采用 Torch/OpenMP 3 个计算线程，为其他系统活动保留余量。

健康检查也是实际干扰源之一。启动 Python 发起一次健康请求，会产生远高于请求本身的 CPU 开销，后续改用轻量原生 curl，减少周期性竞争。这个观察是竞争证据，并不意味着所有抖动都由健康检查引起。

模型执行固定在初始化主线程，通过队列处理请求，维持一致的线程与资源所有权。HTTP 接收可以并发，但模型计算仍串行执行，因此结果不构成并发吞吐承诺。

三线程版正式部署后，100 次 LAN HTTP 的中位数为 859.21ms，P95 为 870.34ms，最慢一次为 875.88ms，建立了经过真实接口验证的亚秒级基线。

代码引用：[三线程服务入口](optimized-deployment/server.py)；后续已部署 graph-half 的同类机制见 [server.py](cpu-tail-deployment/server.py) 中的 `pending` 队列、`queued_answer()` 与主线程执行循环，以及 [run.sh](cpu-tail-deployment/run.sh) 的 CPU 亲和与 curl 健康检查配置。

## 5. 第三阶段：减少 CPU 与 NPU 之间的张量往返

### 5.1 桥接融合、共享准备与独立投影并行

进入亚秒区间后，优化重点转向 CPU 与 NPU 之间的连接部分。

一次矩阵乘的实际成本还包含输入转换、连续化、缓冲区准备、提交与等待，以及输出转换。如果每个投影都独立执行这些步骤，整网会产生大量重复工作。

一次分解中，自有桥接、转换和分配开销已经与 SDK 同步区间处于相近量级。减少一次完整张量往返，可能比继续压缩某个矩阵乘更有效。

我们逐步将 BF16、FP16、FP32 转换、有限值检查和输出整理融合到自有桥接代码中，并让同输入投影共享准备结果。

对于没有数据依赖的分支，例如 MLP 的 gate/up、注意力的 Q/K/V，使用独立上下文和持久工作线程并行执行。可合并的投影进一步合并提交，平衡两个核心的工作，并直接 scatter 到目标位置。经过投影合并，矩阵句柄数量从 78 个减少到 72 个。

并行执行始终遵守一个条件：所有 worker 完成后，才能复用缓冲、传播错误或释放资源。局部失败时，不能将部分成功结果提交给后续网络。

这里缓存的是权重、句柄和工作缓冲，**每次请求仍然执行模型，没有缓存答案。**

代码引用：[adapter.py](bridge-parallel-experiment/adapter.py) 的 `Engine.group()` 共享输入准备并等待所有任务；[fused_bridge.cpp](bridge-parallel-experiment/fused_bridge.cpp) 实现转换融合；[safe_scatter_bridge.cpp](projection-compile-experiment/safe_scatter_bridge.cpp) 对应后续合并与 scatter 路径。

### 5.2 按实际访问顺序优化布局

卷积改为直接读取 BTC 布局，减少转为 BCT 后产生的复制。扫描前的 BF16→FP32 转换、布局整理、Q/K 归一化与缩放被合并成一次遍历；常见数值转换使用 NEON 位操作加速。

这些操作没有改变模型要计算的内容，却减少了内存访问次数和中间结果。

代码引用：[扫描前处理 C 内核](remaining-500ms-experiment/scan_prepare.c)、[前处理接入](remaining-500ms-experiment/prepared_scan.py)。阶段测量与采用范围见 [RESULTS.txt](remaining-500ms-experiment/RESULTS.txt)。

## 6. 第四阶段：优化递推扫描，融合小算子

### 6.1 扫描分块必须保留状态连续性

GatedDeltaNet 扫描经历了多轮实现。最初使用紧凑布局和原生分块递推，随后尝试将相同递推组织成小块矩阵运算，最终在短输入路径中采用 BLAS2 块扫描。

分块只是内部执行方式，FP32 状态必须贯穿所有块，不能在块边界重新归零。是否保留这一点，需要真实层输入回放和边界测试共同证明。

代码引用：[原生扫描适配](scan-optimization-experiment/scan.py)、[BLAS2 块扫描内核](attention-chunk-experiment/chunk_blas2.c)、[对应适配器](attention-chunk-experiment/chunk_blas2.py)。

### 6.2 利用 BF16 值域构建激活查表

随着扫描和大型投影逐渐变快，小算子的累计开销也开始值得处理。后续加入残差相加与归一化融合、注意力准备融合，以及基于 BF16 值域的激活查表。

BF16 只有 65,536 种位模式，可以在初始化时预计算 SiLU、Sigmoid 等结果。推理时通过查表减少重复计算，再把相邻乘法和输出整理融合进去。

下面是实际查表初始化的核心代码：

```python
values=torch.arange(65536,dtype=torch.int32).short().view(torch.bfloat16)
self.silu=F.silu(values).contiguous();self.sigmoid=torch.sigmoid(values).contiguous();self.gate=torch.empty(65536,dtype=torch.float32)
```

查表本身并不足以保证正确。真实张量可能采用不同密集布局，特殊值与舍入边界也需要保持。早期激活查表曾因布局变化导致扩展样本超差，修复后才进入后续候选。

代码引用：[lut_ops.py](cpu-tail-experiment/lut_ops.py) 的 `Tables` 与 `activation()`；[residual.py](cpu-tail-experiment/residual.py) 实现残差与归一化融合；[attention_prepare.py](cpu-tail-experiment/attention_prepare.py) 接入 Q/K norm、RoPE 等准备融合；[candidate.py](cpu-tail-experiment/candidate.py) 管理候选开关与恢复。

### 6.3 短输入方法耗时的阶段性变化

| 历史候选阶段 | 暖态方法中位数 | 状态说明 |
| --- | ---: | --- |
| 桥接融合与投影并行 | 664.76ms | 中间实验版本 |
| 原生扫描优化 | 570.94ms | 中间实验版本 |
| 带范围保护的 FP16 输出 | 512.25ms | 中间实验版本 |
| 布局、扫描前处理与转换优化 | 454.33ms | 中间实验版本 |
| BLAS2 块扫描 | 442.32ms | 中间实验版本 |
| 最终 CPU 融合版本 | 340.40ms | 正式实验候选，后续保留作图回退基础 |
| 最终 graph-half 候选 | 318.51ms | 之后另行验收并部署，HTTP 结果独立统计 |

表中的版本有替换和演进关系，各轮协议也有差异，不能把相邻差值全部归因于单个算子，更不能把各项收益简单相加。

## 7. 已部署版本：固定短输入 MLP 子图与真实 HTTP 验收

到约 340ms 时，我们重新评估了 MLP 子图。

此前尝试过完整注意力 FP16 图。它虽然更快，但扩展内容测试出现超差，没有采用。这个失败并不意味着所有子图都不可用，MLP 可以单独验证。

对于 batch1、115 token 的固定形状，我们捕获原始 12 层 MLP 权重，分别编译 RKNN 子图。第一版采用 FP32 I/O，中间转换较多，整体没有胜过 CPU 融合版。后续直接使用半精度 I/O，减少冗余缓冲，才获得稳定收益。

补充的 20 轮交替对照中，CPU 融合版中位数约 337.93ms，graph-half 约 318.55ms，20 轮全部更快。新进程 100 次测量得到最终方法中位数 318.51ms，P95 328.75ms。

之后，候选经过临时端口验收、正式切换和真实回滚演练，成为 **graph-half v1 已部署版本**。部署后的 100 次 LAN HTTP 中位数为 325.27ms，P95 为 328.92ms。

这次发布使用冻结的运行代码、库和 12 个 MLP 图，保留旧容器用于回退。图快路径只适用于 batch1、115 token，图接口激活形状为 `[1,115,1024]`；其他形状保留 CPU 融合回退路径。不支持的 dtype、训练或梯度模式、权重变化、范围异常等情况也需要旁路或回退。

代码引用：

- [mlp_graph.py](cpu-tail-experiment/mlp_graph.py)：`HalfGraph.run_bf16()` 检查固定输入资格，并调用直接 I/O 接口。
- [mlp_subgraph.py](cpu-tail-experiment/mlp_subgraph.py)：12 层 MLP 子图接入、保护与恢复。
- [runtime.py](cpu-tail-deployment/runtime.py)：持有候选、图与基础模型资源，并在退出时恢复与关闭。
- [config.json](cpu-tail-deployment/config.json)：实际发布的候选配置，避免把筛选中尝试过的开关都算作最终采用项。

验收证据：[部署说明](cpu-tail-deployment/README.md)、[100 次 HTTP 原始记录](cpu-tail-deployment/production-http.json)、[部署汇总](cpu-tail-deployment/deployment-result.json)。

## 8. 未部署实验：把优化扩展到长输入

### 8.1 消除固定形状与旧资格限制带来的性能断崖

短输入进入 300 毫秒级后，下一阶段重点是长输入。**本节及后续 615 token 优化属于独立实验，尚未替换上述已部署版本。**

固定 115 token 图不能直接服务任意长度。最初适配器还有单次矩阵最多 512 行的资格限制，一旦跨过边界，大型投影、注意力或扫描可能回退到较慢的 CPU 路径，形成明显性能断崖。

我们首先利用 MLP 逐 token 独立的特点，新增 128、384 行图，根据实测成本选择组合。`PlannedBank` 对各图进行暖机与计时，再通过动态规划选择覆盖目标行数的组合，而不是固定选择最大桶。

补零只发生在 MLP 内部，额外输出随后丢弃，不改变请求 token、位置或注意力 mask。

随后，将其他大型投影按行分块，每块满足原生接口要求，全部完成后恢复原始行序和列序，再交给注意力或扫描。长序列注意力与扫描也扩展了原生路径：注意力保留完整 K/V，扫描保留跨全部块的连续状态。

代码引用：[planned_mlp.py](mlp-length-experiment/planned_mlp.py) 的 `PlannedBank`、[row_projection.py](row-projection-experiment/row_projection.py)、[long_kernels.py](long-sequence-experiment/long_kernels.py)。适配边界与完整模型实测范围分别见各目录的 `REPRODUCTION.md`，不能将内核覆盖范围等同于所有长度都已完成整网验收。

### 8.2 615 token 注意力与末层按需计算

第五阶段将 QK 与 PV 两次大型矩阵乘交给公开动态 MatMul，保留 CPU FP32 softmax，并融合准备、转换和输出处理。同时，大型投影可以按 615 行一次提交，内部对齐到 640 行。

这一阶段还利用了输出消费者的实际合同：当前 `logits` 最终只使用 encoder 的末位置表示。

因此，在受严格资格保护的私有路径中，前 11 层仍处理全部 615 个位置，第 12 层仍计算完整 615 个 K/V，但 query 和后续逐行 MLP、残差、归一化只计算最后一行。

这没有截断输入，也没有减少最后一行可访问的信息。公开 encoder 仍保留完整 hidden 输出接口；结构、权重版本、方法或钩子变化时，优化路径旁路。发生执行错误时，丢弃私有结果，重新执行原 `logits`，并持续禁用故障快路径。

这里做的是依据消费者合同省去无用中间结果，不是裁掉网络层。这个结论只适用于已检查的末位置读取路径，不能推广到需要完整 hidden 输出的调用。

代码引用：[npu_attention.py](fusion-pipeline-experiment/npu_attention.py) 的 `NpuAttention`；[last_readout.py](fusion-pipeline-experiment/last_readout.py) 的 `good()`、`last_layer()`、`encode_last()` 与 `logits()`。

### 8.3 恢复必要的精度边界

长输入优化暴露了新的数值问题：某些在短输入上合格的 FP16 组合，在新内容或多问题输入上超出原门槛。

因此，第五阶段恢复了必要的 FP32 投影输出与 BF16 MLP 中间边界。恢复 FP32 输出本身并不是新的硬件能力，早期版本就使用过；这次是根据新内容的数值证据重新选择执行精度。

恢复边界后，第五阶段通过 34 个回归用例，615 token 正式实验 P50 为 1745.16ms，P95 为 1776.55ms。此前多问题输入中约 0.01745 的 score 超差也得到修复。

代码与配置引用：[precise_mlp.py](fusion-pipeline-experiment/precise_mlp.py)、[mixed_bridge.cpp](fusion-pipeline-experiment/mixed_bridge.cpp)、[execution.json](fusion-pipeline-experiment/execution.json)。证据见 [第五阶段汇总](fusion-pipeline-experiment/logs/summary.json)。

## 9. 最新未部署实验：保留 BF16 边界，融合 native MLP 数据流

### 9.1 从完整张量往返，改为直接写入后续缓冲

典型 MLP 可以写成：

```text
gate    = W_gate(x)
up      = W_up(x)
product = SiLU(gate) * up
y       = W_down(product)
```

如果 gate/up 的结果先完整解包为普通张量，再经过激活、乘法、转换和重新打包，就会产生额外内存往返。

第六阶段使用公开的 FP16×FP16→FP32 MatMul，直接读取 gate/up 的 native 输出布局，在自有代码中完成原有 BF16 舍入、SiLU 查表和乘法舍入，再写入 down 的 native 输入缓冲。

实际 C++ 代码明确选择了输出类型和布局：

```cpp
rknn_matmul_info info{};info.M=640;info.K=k;info.N=n;info.type=RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;info.AC_layout=RKNN_MM_LAYOUT_NATIVE;info.B_layout=RKNN_MM_LAYOUT_NATIVE;
```

代码会查询并校验 SDK 实际返回的布局，而不是沿用旧 FP16 输出布局的假设。关键转换与舍入逻辑集中在 `nm_fuse_slice()`，范围检查仍然保留。

代码引用：[native_split.cpp](bf16-pipeline-experiment/native_split.cpp) 的 `nm_create()`、`nm_fuse_slice()` 和 `nm_output()`。

### 9.2 双核分担 down 投影，管理长度切换资源

down 投影进一步拆成 320 行和 295 行，交给两个独立上下文执行，全部成功后按原顺序拼回 615 行。实际分块选择为：

```python
parts=[(0,320),(320,295)] if rows==615 else [(0,128),(128,128)];outputs=[]
```

这个底层组件还包含其他实验行数分支，但最终候选的对外资格由上层限制为 B1/T615 `logits`，不能据此宣称其他形状已采用相同优化。

长度切换时，还会退役当前路径不再需要的旧半精度或旧 FP32 MLP 对照上下文，保留有效的暖态资源，避免多个执行版本同时驻留造成资源压力。退役发生在请求边界，前一次调用的 worker 已经结束。

代码引用：[split_mlp.py](bf16-pipeline-experiment/split_mlp.py) 的 `SplitMlp.run()`；[cache_owner.py](bf16-pipeline-experiment/cache_owner.py) 的长度域切换与 `retire()`；最终入口为 [accepted_owner.py](bf16-pipeline-experiment/accepted_owner.py)，采用范围冻结在 [selection.json](bf16-pipeline-experiment/selection.json)。

### 9.3 正式结果：615 token 约 1.58 秒，仍未达到 1 秒目标

最终正式测量采用三轮独立容器，每轮 30 组暖态交错配对：

| 第六阶段同轮对照 | P50 | P95 |
| --- | ---: | ---: |
| 上一合格候选 | 1743.55ms | 1770.60ms |
| 最新候选 | **1579.62ms** | **1642.71ms** |

P50 下降 **9.40%**，90／90 组候选更快。对照数值与第五阶段原测量略有差别，来自独立运行，不能混合统计。

**这一结果尚未部署，也没有达到最初 P50、P95 均低于 1 秒的目标。**

证据入口：[最新正式汇总](bf16-pipeline-experiment/logs/summary.json)、[验证命令与结果](bf16-pipeline-experiment/VERIFY.txt)、[复现说明](bf16-pipeline-experiment/REPRODUCTION.md)。

## 10. 没有进入最终方案的路线，同样值得记录

整个过程中，有不少看起来合理、最终没有采用的方案：

- **完整注意力 FP16 图**：初筛更快，扩展内容数值超差，拒绝采用。
- **部分 CPU 编译方案**：没有建立端到端收益，保留原实现。
- **公开 GPU 路线**：当前实现没有胜过 CPU/NPU 混合路径，未采用；结果不代表 GPU 理论极限。
- **部分 native 布局和转换查表尝试**：减少某一步操作，整体却更慢，未采用。
- **跨层流水与下一层投影预取**：没有稳定胜过最新合格路径，未并入最终候选。

这些失败帮助我们区分三件事：接口支持一种机制、局部算子更快、完整模型更快。只有最后一项经过数值与资源验证后，才有资格进入最终候选。

历史路线见 [全过程 HTML 记录](d3-edge-development-history.html)；该记录截止 graph-half 发布，后续六轮以 [多长度报告](mlp-length-experiment/report.html) 和各实验原始记录为准。旧历史 HTML 的生成检查曾提示需要重新生成，引用具体数值时应以对应原始 JSON 为准。

## 11. 怎样证明更快，同时控制数值偏差

### 11.1 分开统计方法、HTTP、冷启动和组件剖析

测量方法随着优化逐步收敛：正式性能与插桩剖析分开，候选和基线交错运行，独立进程重复测量，保留所有慢样本。

组件时间不直接相加：嵌套计时可能包含子项，并行 worker 可能重叠，各项中位数之和也不等于总中位数。SDK 调用区间不称为纯 NPU 计算时间，因为其中还包含提交、驱动和等待。

方法调用不包含 HTTP，HTTP 单请求延迟也不能直接转换为并发吞吐或硬延迟上界。初始化和首次形状准备成本另外记录，不混入暖态成绩。

### 11.2 数值回归不等于业务准确率评测

最新 34 个回归用例中，choice 全部一致。相对冻结的 current 参考执行路径，最大绝对差为：

| 字段 | 最新候选最大绝对差 | 验证口径 |
| --- | ---: | --- |
| 分类概率 | 0.009295 | 原门槛 ≤ 0.01，通过 |
| score | 0.007060 | 原门槛 ≤ 0.01，通过；score 范围为 0–2 |
| noul | 0.003011 | 原门槛 ≤ 0.01，通过 |
| confidence | 0.014146 | 独立报告，不包含在上述通过结论中 |

这里的 current 是实验冻结的参考执行路径，不应笼统写成“所有官方实现都严格一致”。相对上一合格候选，最新版本的最大受检数值差约为 `4.74e-7`。

已部署 graph-half 的历史回归还应单独理解：原比较器没有覆盖 confidence，审计后独立披露其相对 442ms 候选的最大差约 0.010027。HTTP 与冻结 graph-half 候选的等价性检查则包含 confidence，二者参考对象不同，不能混用。

这些证据支持“已测样本通过数值回归”，还不足以证明所有业务数据上的准确率完全无损。要做出后一种结论，仍需要标注业务数据集评测。

验证代码引用：[validate_records.py](bf16-pipeline-experiment/validate_records.py) 中的 `compare()` 与 `validate()`；已部署版本的独立审计见 [confidence-audit.json](cpu-tail-deployment/confidence-audit.json)，HTTP 校验见 [verify_http.py](cpu-tail-deployment/verify_http.py)。

### 11.3 性能之外，也验证故障与生命周期

快路径必须检查输入范围、有限值、形状、训练模式、权重版本和钩子等条件。分块或并行执行发生错误时，需要等待所有 worker，再丢弃私有结果并按相应范围回退，避免让部分结果进入后续层。

验证还覆盖构造失败清理、长度切换、句柄与 FD 暖态稳定、关闭与重复关闭。独立实验期间保留已部署服务，并核对容器身份、配置、发布文件哈希和固定 HTTP 输出。

最新候选仍未覆盖任意业务输入、多 Owner 并发、冷启动 SLA 或并发吞吐；这些属于后续发布验证范围。

## 12. 从这轮优化得到的工程经验

这次工作的主线可以概括为：

> 保留原模型权重、12 层结构和关键计算语义，通过优化推理引擎的计算分工、算子实现、数据流与调度降低延迟，并用回归测试控制数值偏差。

每个阶段都需要重新定位主导成本：从 CPU BF16 算子，到大型矩阵乘，再到卷积、扫描、布局复制、dtype 转换和请求调度。随着模型变快，原本不起眼的开销逐渐成为下一轮瓶颈。

目前，已部署版本完成了固定短输入约 325ms 的真实 HTTP 验证；最新实验候选将 615 token 暖态方法调用推进到约 1.58s。后续继续优化，需要针对剩余瓶颈重新取证，也需要把实验候选补齐为真正可发布、可回退的服务版本。

## 附录：复现环境与阅读入口

留存测试环境为 RK3576、16GB 内存、ARM64 Debian 13，固定 CPU 4–7；已部署版本使用 Torch/OpenMP 3 线程，最新长输入实验的扫描路径另使用 4 线程。模型快照为 `20c745653400d191ecacd99c4eafc0c5900dc15e`，RKNN Runtime 2.5.0、驱动 0.9.8。更换环境后不能直接沿用本文性能结论。

模型、ARM 库、RKNN 图与厂商 SDK 二进制不随 Git 提交。复现需要按各阶段说明准备相应工件，克隆源码并不等于具备完整可运行环境。

| 阅读目的 | 入口 |
| --- | --- |
| 当前状态与已部署/实验边界 | [WORK_STATUS.md](WORK_STATUS.md) |
| CPU 到 graph-half 的历史演进 | [全过程记录](d3-edge-development-history.html)、[生成器与阶段索引](build_development_history.py) |
| 已部署版本的构建依赖与升级步骤 | [部署说明](cpu-tail-deployment/README.md)、[318ms 实验复现](cpu-tail-experiment/REPRODUCTION.txt) |
| 已部署版本的 HTTP 证据 | [部署汇总](cpu-tail-deployment/deployment-result.json)、[原始 HTTP 记录](cpu-tail-deployment/production-http.json) |
| 多长度六轮实验总览 | [多长度报告](mlp-length-experiment/report.html) |
| 最新候选实现与冻结策略 | [accepted_owner.py](bf16-pipeline-experiment/accepted_owner.py)、[selection.json](bf16-pipeline-experiment/selection.json) |
| 最新实验结果与验证 | [summary.json](bf16-pipeline-experiment/logs/summary.json)、[VERIFY.txt](bf16-pipeline-experiment/VERIFY.txt) |

发布前如需更新数值或部署状态，应同时核对对应原始记录与实际发布版本，继续区分短输入/长输入、方法/HTTP、已部署/实验候选，以及数值回归/业务准确率。
