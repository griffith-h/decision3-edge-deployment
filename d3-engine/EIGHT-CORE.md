# RK3576 八核跑 d3.1-nano 的实测

在 16 GB NanoPi R76S 上用同一份 nano 检查点跑 13 条公开请求。样本包括 TypeSafe WorkflowEvals 的客服与 55 题发票、vLLM 发布结果中的三条完整 Decision Index 请求，以及 vLLM 中文路由留出集的八条固定抽样。保留原文、题序、选项、FP32 决策头和官方策略代码。[数据版本与抽样方法](README.md) · [芯片规格](https://www.rock-chips.com/uploads/pdf/2024.3.18/192/RK3576%20Brief%20Datasheet.pdf)

**目前整张发票最快的是八核 CPU 配合双核 RKNPU。** 八核模式 `--aggressive` 默认将至少有 2,048 个键、16 个查询位置且形状受支持的注意力段送入 RKNN。相对本轮旧引擎复测，整单从 20 分 19 秒缩短到 **11 分 20 秒，减少 44.3%**，离 10 分钟还差 79.69 秒。订单则从 20.36 秒降至 14.86 秒。[板端环境、检查点和库哈希](results/eight-core/hardware.json) · [两轮长单原始记录](results/eight-core/long8.json)

## 八核如何参与推理

四个 Cortex-A72 和四个 Cortex-A53 共同运行 Torch、卷积、递推和数据转换。两个 RKNPU 核心可以并行执行 MLP 的 gate、up 投影，也可以各自负责长注意力的一个 K/V 头。每个请求只拥有一份公共状态 K/V，最后一层每题只计算决策读出位置。单问题、单序的请求一次通过主干。长注意力按有效键分块做在线 softmax，短段由 ARM 完成。[前缀注意力](fast_prefix.py) · [RKNN 分块实现](prefix_segment.cpp)

同一张发票中，状态有 8,902 个 token，55 个问题合计 26,092 个 token。旧引擎本轮热调用 1,219.25 秒，返回值与历史的 1,225.57 和 1,234.16 秒两轮逐项相同。八核版保留 ARM 注意力时用 746.92 秒，这一轮还保存了两份张量供后续重放。相同请求的投影和 MLP 矩阵乘提交数从 44,250 减少到 22,115。末层跳过 8,902 个状态查询，保留 55 个决策查询。[旧版同板复测](results/eight-core/reference.json) · [八核 ARM 注意力和采样](results/eight-core/arm8-capture.json)

长段交给双核 RKNN 后，两次正式发票热调用为 678.94 和 680.44 秒，中位数 679.69 秒。两次输出逐项相同。进程峰值 RSS 8.33 GiB，容器峰值 5.60 GiB，交换空间用量为零。第一轮 80 次 RKNN 长注意力共用 233.49 秒，345 次 ARM 短段和末位查询共用 24.53 秒，投影和 MLP 连同转换、搬运与同步共用 221.98 秒。紧凑状态 K/V 合计 104.32 MiB。[两轮完整返回与计数](results/eight-core/long8.json)

| 55 题整单 | 实测热延迟 | 进程平均占用核心数 | 八核同时超过或达到 90% 的一秒采样窗口 |
| --- | ---: | ---: | ---: |
| 旧引擎，CPU 4–7，Torch 3 线程，RKNN 投影 | 1,219.25 s | 未采样 | 未采样 |
| 八核新引擎，ARM 长注意力，保存重放张量 | 746.92 s | 5.17 | 122 / 745 |
| 八核新引擎，RKNN 长注意力 | 678.94、680.44 s | 3.10 | 0 / 1,356 |

核心数由模型进程的 CPU 时间除以墙钟时间计算。最快版本的单窗口峰值是 699.2%，八个逻辑 CPU 均有工作。ARM 长注意力让更多 CPU 周期参与矩阵运算，交给 RKNPU 后整单又少用约 67 秒。RSS 包含映射权重，cgroup 峰值按容器记账，两个指标各自报告。[CPU、内存与温度采样](bench.py)

四个 A72 与全部八核使用同一新版引擎的整单对照仍缺结果。在板端的 Mali 长矩阵诊断期间连接中断，未取回该项可核验的计时。当前结果支持 RKNN 长段和 ARM 短段的组合。Mali 在 nano 长输入上的速度和四个 A53 单独带来的收益仍待实测。Rockchip 芯片框图列出双簇 CPU、Mali-G52 MC3 和 RKNPU，未列独立 TPU。[Rockchip RK3576 框图](https://www.rock-chips.com/uploads/pdf/2024.3.18/192/RK3576%20Brief%20Datasheet.pdf)

| 发票中的完整片段 | 八核 ARM FP32，两轮 | 双核 RKNN + ARM，两轮 | 对 ARM 的最大输出绝对差 |
| --- | ---: | ---: | ---: |
| 8,902 个状态位置互相读取 | 14.351 / 14.429 s | 12.338 / 11.633 s | 0.000612 |
| 4 个问题各 130 个位置读取同一段 8,902-token 状态 | 1.116 / 1.126 s | 0.719 / 0.701 s | 0.000291 |

两组分别按 ARM、RKNN、RKNN、ARM 顺序重放同层的真实 Q/K/V。计时覆盖布局转换、设备调用、同步和输出恢复。表中误差属于该层的浮点输出，最终选项概率另行核对。[张量形状、哈希与逐次测量](results/eight-core/prefix-probe.json)

## 业务输出

12 条有纯 CPU 对照的请求共 58 个字段、172 个选项。八核计算和原版纯 CPU 在 58 个字段上的最终判断一致。170/172 个选项的概率相差不超过 1 个百分点。家电亮度题的两个选项分别由 63.1534% 变为 61.8677%、36.5583% 变为 37.8478%。Score 的最大期望分差为 0.003915 分。[逐选项概率与差值](results/eight-core/suite8-cpu.json)

55 题发票的概率参照来自旧版同板 RKNN，本张发票缺少原版纯 CPU 整单对照。八核 RKNN 长注意力与旧版的 55 个字段判断全部一致，170/171 个选项的差异在 1 个百分点内。最大差来自第一项发票行的 `extra_unapproved`，旧版给 71.4882%，新版给 70.3299%。对 TypeSafe 共识，55 题命中 45 个，三个策略都把疑似重复列为首要动作，官方首要动作是争议款项复核。[发票逐选项差异](results/eight-core/long8-vs-previous.json) · [官方三策略重放](results/eight-core/long8-policies.json)

这些样本的八核热态中位数分别是订单 14.86 秒、家电 39.05 秒、合同 50.62 秒、TypeSafe 客服 23.22 秒，八条 vLLM 中文路由为每条 2.59–4.44 秒。短套件显式指定 ARM 注意力，投影与 MLP 使用 RKNN。12 条中最长的公共段只有 1,939 个键，均走八核模式的短段分支。默认 RKNN 长注意力配置复跑订单为 14.91 秒，答案与单独的短段基准逐项一致。每条短请求跑两次，订单首轮含权重打包，热态只取第二轮。对 12 条逐条计算，相对同板原版纯 CPU 的加速在 66.9–107.5 倍之间。客服命中 9/11 个共识标签，三套策略均返回一般咨询回复，官方期望账户恢复。中文工具判断命中 2/4，反馈分类命中 3/4。[八核完整响应](results/eight-core/suite8.json) · [客服策略重放](results/eight-core/customer-policies.json)

## 复测

在普通电脑的仓库根目录运行 `python3 -B d3-engine/verify_results.py` 即可重算七份概率对照，包括本轮 12 条 CPU 配对和 55 题发票配对。CPU/NPU 分担同一段查询、投影合并和 Mali 矩阵乘另存有原型源码，目前缺少这些方案的整单板测结果。[离线复核脚本](verify_results.py) · [原型源码与验证范围](experiments/long-prefix/README.md)

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

RK3576 固定镜像执行 `python -m unittest -v test_engine`，当时的 13 项均通过，包含真实 NPU 上跨 8,192 个键的矩形注意力、256 行投影尾块和数值越界回退。发布前在相同 Torch 和 Transformers 版本重跑 10 项可移植测试，覆盖最终的默认后端、单题模式退出和并发锁。宿主机执行 `python3 -B d3-engine/test_compare.py` 六项通过，ShellCheck 和 `git diff --check` 通过。两份新对照报告均可从原始文件逐项重建。四核整单和最后的容器退出状态在板子恢复连接后方可核对。[引擎测试](test_engine.py) · [比较器测试](test_compare.py)
