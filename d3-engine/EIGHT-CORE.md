# RK3576 八核跑 d3.1-nano 的实测

在 16 GB NanoPi R76S 上用同一份 nano 检查点跑 13 条公开请求。样本包括 TypeSafe WorkflowEvals 的客服与 55 题发票、vLLM 发布结果中的三条完整 Decision Index 请求，以及 vLLM 中文路由留出集的八条固定抽样。保留原文、题序、选项、FP32 决策头和官方策略代码。[数据版本与抽样方法](README.md) · [芯片规格](https://www.rock-chips.com/uploads/pdf/2024.3.18/192/RK3576%20Brief%20Datasheet.pdf)

**八核 CPU 配合双核 RKNPU，最快实验配置两次跑完 55 题发票分别用时 598.32 和 599.21 秒，中位数 598.76 秒。** 同次八核正式版为 679.15 秒，最初的同板引擎为 1,219.25 秒。三个版本对 55 个字段的判断一致。最快配置与当前正式版比较时，有两个选项超过 1 个百分点。要求每个选项都在这个范围内时，实测中位数最快的是 601.03 秒的配置。[板端环境与库哈希](results/long-prefix/platform.json) · [最快配置与正式版的差值](results/long-prefix/shapes-vs-eight.json) · [601.03 秒配置的差值](results/long-prefix/padded-vs-eight.json)

## 八核如何参与推理

四个 Cortex-A72 和四个 Cortex-A53 共同运行 Torch、卷积、递推和数据转换。两个 RKNPU 核心可以并行执行 MLP 的 gate、up 投影，也可以各自负责长注意力的一个 K/V 头。每个请求只拥有一份公共状态 K/V，最后一层每题只计算决策读出位置。单问题、单序的请求一次通过主干。长注意力按有效键分块做在线 softmax，短段由 ARM 完成。[前缀注意力](fast_prefix.py) · [RKNN 分块实现](prefix_segment.cpp)

同一张发票中，状态有 8,902 个 token，55 个问题合计 26,092 个 token。旧引擎本轮热调用 1,219.25 秒，返回值与历史的 1,225.57 和 1,234.16 秒两轮逐项相同。八核版保留 ARM 注意力时用 746.92 秒，这一轮还保存了两份张量供后续重放。相同请求的投影和 MLP 矩阵乘提交数从 44,250 减少到 22,115。末层跳过 8,902 个状态查询，保留 55 个决策查询。[旧版同板复测](results/eight-core/reference.json) · [八核 ARM 注意力和采样](results/eight-core/arm8-capture.json)

长段交给双核 RKNN 后，两次正式发票热调用为 678.94 和 680.44 秒，中位数 679.69 秒。两次输出逐项相同。进程峰值 RSS 8.33 GiB，容器峰值 5.60 GiB，交换空间用量为零。第一轮 80 次 RKNN 长注意力共用 233.49 秒，345 次 ARM 短段和末位查询共用 24.53 秒，投影和 MLP 连同转换、搬运与同步共用 221.98 秒。紧凑状态 K/V 合计 104.32 MiB。[两轮完整返回与计数](results/eight-core/long8.json)

| 55 题整单 | 实测热延迟 | 进程平均占用核心数 | 八核同时超过或达到 90% 的一秒采样窗口 |
| --- | ---: | ---: | ---: |
| 旧引擎，CPU 4–7，Torch 3 线程，RKNN 投影 | 1,219.25 s | 未采样 | 未采样 |
| 八核新引擎，ARM 长注意力，保存重放张量 | 746.92 s | 5.17 | 122 / 745 |
| 八核新引擎，RKNN 长注意力 | 678.94、680.44 s | 3.10 | 0 / 1,356 |

核心数由模型进程的 CPU 时间除以墙钟时间计算。这一轮 RKNN 长注意力的单窗口峰值是 699.2%，八个逻辑 CPU 均有工作。ARM 长注意力让更多 CPU 周期参与矩阵运算，交给 RKNPU 后整单又少用约 67 秒。RSS 包含映射权重，cgroup 峰值按容器记账。[CPU、内存与温度采样](bench.py)

### 同次启动的四核和八核对照

2026-11-20 恢复连接后，先用同一条官方订单预热，再在各自的独立容器内各跑两次同一张 55 题发票。两个 RKNPU 核、镜像、权重、引擎和每组问题数相同。四核组开放 CPU 4–7、Torch 用 4 线程，八核组开放 CPU 0–7、Torch 用 8 线程，两组 OpenBLAS 均为 1 线程。[本轮环境、源码与权重哈希](results/long-prefix/platform.json)

| CPU 与双核 RKNPU 的分工 | 两次发票热延迟 | 中位数 | 进程平均使用核心数 | 峰值 RSS / cgroup |
| --- | --- | ---: | ---: | ---: |
| 4 × A72 | 731.77 / 732.12 s | 731.95 s | 1.99 | 8.33 / 5.62 GiB |
| 4 × A72 + 4 × A53 | 676.07 / 682.24 s | 679.15 s | 3.11 | 8.33 / 5.60 GiB |
| 八核，CPU 分担 3/8 长查询行 | 619.41 / 619.86 s | 619.63 s | 4.51 | 8.44 / 5.72 GiB |
| 分担查询并合并投影 | 607.14 / 607.00 s | 607.07 s | 4.51 | 9.11 / 6.38 GiB |
| 合并投影并减少分批填充 | 602.07 / 599.98 s | 601.03 s | 4.51 | 9.13 / 6.40 GiB |
| 在上一组基础上，CPU 分担一半方形长段查询行 | 598.32 / 599.21 s | **598.76 s** | 4.61 | 9.14 / 6.41 GiB |

开放四个 A53 后，中位数缩短 52.80 秒（7.21%）。55 个字段判断一致，171 个选项均在 1 个百分点内，最大差 0.1745 个百分点。此次八核相对上次八核的 171 个选项概率逐项相同。两组矩阵乘提交均为每单 22,115 次，交换空间用量为零。[四核原始响应](results/long-prefix/four.json) · [八核原始响应](results/long-prefix/eight.json) · [逐选项比较](results/long-prefix/four-vs-eight.json)

四个 A72 配双核 NPU 的 12 条短请求也各跑了两次。订单、家电、合同和客服的热态分别为 15.78、42.11、56.37、24.45 秒，八条中文路由每条 2.67–4.47 秒。与八核正式版逐项比较，58 个判断和 172 个概率完全相同。与原版纯 CPU 比较，58 个判断相同，170/172 个选项在 1 个百分点内。[四核完整返回](results/long-prefix/four-suite.json) · [逐选项 CPU 对照](results/long-prefix/four-suite-cpu.json)

同一启动中还用全部八核、8 个 Torch 线程直接运行了原版纯 CPU 检查点。vLLM 中文反馈 `fresh-crosswoz-000261` 共 218 个输入 token。纯 CPU 两次为 135.16 / 135.08 秒，第二轮平均占 6.78 核，峰值 RSS / cgroup 为 3.31 / 0.66 GiB。历史 3 线程纯 CPU 用时 197.23 秒，两种线程配置的五个概率完全一致。此次四个大核配双核 NPU 的热态中位数为 2.74 秒，相对八核纯 CPU 第二轮快 49.27 倍。两者都命中官方标签 `SAT`，其概率由 96.8764% 变为 96.8061%，五个选项的最大绝对差为 0.0703 个百分点。[八核纯 CPU 原始响应](results/long-prefix/cpu-eight.json) · [三线程纯 CPU 参照](results/router-cpu.json) · [同板四核加速对照](results/long-prefix/four-suite-cpu.json)

### CPU 与 NPU 同时计算长注意力

实验版显式指定 `--aggressive --prefix-backend hybrid`。在受支持且至少有 64 行查询的长段上，八核 ARM 计算约 3/8 的查询行，两个 NPU 核同时处理其余行。每行都访问相同的完整 K/V。两次整单为 **619.41 和 619.86 秒，中位数 619.63 秒**。相对同次启动的八核 RKNN 对照少了 59.52 秒（8.76%）。80 个长段的总耗时从 230.99 / 235.99 秒降到 177.97 / 176.16 秒。[实验配置与计数](results/long-prefix/hybrid.json) · [张量微基准](experiments/long-prefix/README.md#真实张量重放)

模型进程平均占用 4.51 个核心。两轮分别有 30/618 和 33/618 个一秒窗口内，八核忙碌率同时达到 90%。峰值 RSS 为 8.44 GiB，容器峰值为 5.72 GiB，交换空间读数为零。相对同次八核 RKNN 对照，55 个判断一致，171/171 个选项的差值在 1 个百分点内，最大 0.8526 个百分点。相对更早的旧引擎，55 个判断仍一致，170/171 个选项在门槛内，最大差 1.4949 个百分点。TypeSafe 官方策略代码重放得到的三个首要动作及动作集合与上次相同，完整报告逐字节相同。[当前八核对照](results/long-prefix/hybrid-vs-eight.json) · [旧引擎对照](results/long-prefix/hybrid-vs-previous.json) · [官方策略重放](results/eight-core/long8-policies.json)

在同一实验引擎上打开 `--fuse-projections`，相邻投影共用输入，每张发票合并 384 批投影，矩阵乘提交从 22,115 次降至 17,707 次。两轮热延迟为 **607.14 / 607.00 秒，中位数 607.07 秒**，比只分担查询少 12.56 秒。峰值 RSS / cgroup 增至 9.11 / 6.38 GiB。模型进程平均占用 4.51 个核心，两轮合计有 76/1,211 个一秒窗口八核同时达到 90%。与只分担查询的版本相比，55 个判断、171 个概率和官方策略重放结果逐项相同。[合并投影的完整记录](results/long-prefix/combined.json) · [逐选项比较](results/long-prefix/combined-vs-hybrid.json)

在这套配置上按题长调整同一批内的问题数量，55 题仍按原有顺序分 15 批，每批最多 4 题。实际计算的问题位置从 27,771 减为 27,288，矩阵乘提交再降至 17,475 次。两轮为 **602.07 / 599.98 秒，中位数 601.03 秒**，峰值 RSS / cgroup 为 9.13 / 6.40 GiB。相对上一步，所有判断及官方策略相同，最大选项概率差为 0.5117 个百分点。[实测原始响应](results/long-prefix/padded.json) · [概率差值](results/long-prefix/padded-vs-combined.json)

最后只调整单序列且查询数等于键数的长段，CPU 分担比例提高到 1/2，其他长段继续按 3/8。每单 80 个长段中有 5 个走了 1/2，实测中位数比固定 3/8 再降 2.26 秒。共 1,195 个一秒窗口中有 102 个八核同时达到 90% 忙碌率，模型进程平均占用 4.61 核，峰值 RSS / cgroup 为 9.14 / 6.41 GiB。与同次八核正式版相比，第一、二项发票行的 `extra_unapproved` 分别从 70.3299% 和 47.3517% 变为 71.5885% 和 48.4389%，其余 169 个选项在 1 个百分点内。与最初的同板引擎相比，171 个选项全部在门槛内，最大差 0.9269 个百分点。55 题仍命中 45 个官方共识标签，三套策略的重放报告与最初结果逐字节相同。[最快配置的两次完整响应](results/long-prefix/shapes.json) · [对固定 3/8](results/long-prefix/shapes-vs-padded.json) · [对当前八核版](results/long-prefix/shapes-vs-eight.json) · [对最初引擎](results/long-prefix/shapes-vs-previous.json) · [官方策略报告](results/eight-core/long8-policies.json)

芯片框图列出了双簇 CPU、Mali-G52 MC3 和 RKNPU，未列独立 TPU。[Rockchip RK3576 框图](https://www.rock-chips.com/uploads/pdf/2024.3.18/192/RK3576%20Brief%20Datasheet.pdf)

| 发票中的完整片段 | 八核 ARM FP32，两轮 | 双核 RKNN + ARM，两轮 | 对 ARM 的最大输出绝对差 |
| --- | ---: | ---: | ---: |
| 8,902 个状态位置互相读取 | 14.351 / 14.429 s | 12.338 / 11.633 s | 0.000612 |
| 4 个问题各 130 个位置读取同一段 8,902-token 状态 | 1.116 / 1.126 s | 0.719 / 0.701 s | 0.000291 |

两组分别按 ARM、RKNN、RKNN、ARM 顺序重放同层的真实 Q/K/V。计时覆盖布局转换、设备调用、同步和输出恢复。表中误差属于该层的浮点输出，最终选项概率另行核对。[张量形状、哈希与逐次测量](results/eight-core/prefix-probe.json)

Mali-G52 r1（Panfrost）在完全相同的问题段上只计算 FP32 QKᵀ。输入驻留 GPU 缓冲区，分 153 块覆盖全部分数，输出缓冲区限为 1 MiB。第一次提交与同步累计 5.270 秒，两轮热态为 **2.533 / 2.531 秒**，计入块内复制和抽查后为 3.709 / 3.714 秒。每轮抽查 1,224 个分数，最大绝对误差为 0.00000329。对这同一段，八核 CPU 配双核 NPU 完成 QKᵀ、softmax 和乘 V 合计只用 **0.601 / 0.619 秒**。据此，整单保留 CPU 与 RKNPU 的组合。GPU 记录的是这套分块 CLBlast 实现的算子用时。[GPU 三轮原始计时](results/long-prefix/mali.jsonl) · [完整注意力的配对张量](results/long-prefix/query-split-probe.json)

## 业务输出

最快实验配置在另 12 条官方请求上各跑两次。订单、家电、合同、TypeSafe 客服的热态中位数分别为 14.67、39.01、49.79、22.51 秒，八条 vLLM 中文路由分别为 2.60–4.38 秒。与同板原版纯 CPU 比较，58/58 个决策一致，170/172 个选项在 1 个百分点内，最大差 1.2894 个百分点。两项超限仍属于同一道家电亮度题。客服命中 9/11 个共识标签，三个官方策略的重放报告与正式版逐字节相同。218-token 的中文反馈在此配置下为 2.64 秒，比同次八核纯 CPU 的热调用 135.08 秒快 51.23 倍，全部五个选项最大相差 0.0703 个百分点。[12 条完整返回](results/long-prefix/shapes-suite.json) · [172 个选项的纯 CPU 对照](results/long-prefix/shapes-suite-cpu.json) · [同次八核纯 CPU 对照](results/long-prefix/shapes-suite-cpu-eight-feedback.json) · [官方客服策略](results/eight-core/customer-policies.json)

八核正式版和原版纯 CPU 在这 58 个字段上的最终判断也一致。170/172 个选项的概率相差不超过 1 个百分点。家电亮度题的两个选项分别由 63.1534% 变为 61.8677%、36.5583% 变为 37.8478%。Score 的最大期望分差为 0.003915 分。[正式版逐选项概率与差值](results/eight-core/suite8-cpu.json)

55 题发票的概率参照来自旧版同板 RKNN，本张发票缺少原版纯 CPU 整单对照。八核 RKNN 长注意力与旧版的 55 个字段判断全部一致，170/171 个选项的差异在 1 个百分点内。最大差来自第一项发票行的 `extra_unapproved`，旧版给 71.4882%，新版给 70.3299%。对 TypeSafe 共识，55 题命中 45 个，三个策略都把疑似重复列为首要动作，官方首要动作是争议款项复核。[发票逐选项差异](results/eight-core/long8-vs-previous.json) · [官方三策略重放](results/eight-core/long8-policies.json)

这些样本的八核热态中位数分别是订单 14.86 秒、家电 39.05 秒、合同 50.62 秒、TypeSafe 客服 23.22 秒，八条 vLLM 中文路由为每条 2.59–4.44 秒。短套件显式指定 ARM 注意力，投影与 MLP 使用 RKNN。12 条中最长的公共段只有 1,939 个键，均走八核模式的短段分支。默认 RKNN 长注意力配置复跑订单为 14.91 秒，答案与单独的短段基准逐项一致。每条短请求跑两次，订单首轮含权重打包，热态只取第二轮。对 12 条逐条计算，相对同板原版纯 CPU 的加速在 66.9–107.5 倍之间。客服命中 9/11 个共识标签，三套策略均返回一般咨询回复，官方期望账户恢复。中文工具判断命中 2/4，反馈分类命中 3/4。[八核完整响应](results/eight-core/suite8.json) · [客服策略重放](results/eight-core/customer-policies.json)

## 复测

在普通电脑的仓库根目录运行 `python3 -B d3-engine/verify_results.py` 可重算归档的概率对照，包括 12 条纯 CPU 配对和 55 题发票的引擎间配对。每个报告保留全部逐选项差值和原始门槛结论。[离线复核脚本](verify_results.py) · [原型源码与验证范围](experiments/long-prefix/README.md)

先按项目说明生成官方请求、准备固定权重并运行 `stage-board.sh`。以下命令在板上创建独立容器，模型加载单独计时。订单先给惰性权重打包预热。基准按请求保存完整返回值、输入哈希、内存、温度、进程 CPU 时间和各核忙碌率。[环境与公开数据准备](README.md#板端复测)

```bash
invoice=( "$lab"/src/typesafe-invoice-*.json )
cases=( /bench/order.json "/bench/$(basename "${invoice[0]}")" )
bash "$lab/src/run-board.sh" "$lab" "$checkpoint" "$run_id-eight" rknn \
    /out/eight.json "${cases[@]}" --aggressive --prefix-backend rknn --repeats 2
D3_CPUS=4-7 D3_THREADS=4 bash "$lab/src/run-board.sh" "$lab" "$checkpoint" "$run_id-four" rknn \
    /out/four.json "${cases[@]}" --aggressive --prefix-backend rknn --repeats 2
```

CPU 与八核的线程配置不同，比较时显式记录两边的 CPU 集合和线程数。该命令能重建归档的 172 项对照，亮度题超出 1 个百分点时返回状态码 1。

```bash
d=d3-engine/results
python3 -B d3-engine/compare.py "$d/cpu.json" "$d/eight-core/suite8.json" \
    --also-cpu "$d/router-cpu.json" --also-cpu "$d/workflow-cpu.json" --compare-cpu-schedules
python3 -B d3-engine/compare.py "$d/eight-core/reference.json" "$d/eight-core/long8.json" \
    --engine-request typesafe-invoice-processing-ap_00000-semantic-c99c14de6091
```

RK3576 固定镜像执行 `python -B -m unittest -v test_engine`，本轮正式版 14 项、基础实验版 15 项、调整分批后的 15 项和按形状分工后的 16 项全部通过。覆盖真实 NPU 的长段、投影尾块、数值越界回退、融合投影更新权重、填充和查询行的分配。宿主机比较器的六项回归由 `python3 -B d3-engine/test_compare.py` 执行。[板端版本与测试记录](results/long-prefix/platform.json) · [引擎测试](test_engine.py) · [比较器测试](test_compare.py)
