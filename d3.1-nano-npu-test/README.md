# d3.1-nano 在 RK3576 上的 CPU 与 NPU 实测

2026-11-19，同一条 115-token 中文分类请求在 NanoPi R76S 上用原版纯 CPU 推理需要 **108.09 秒**。将文本矩阵乘迁到 NPU 后，热态耗时为 **10.87 秒**。针对单问题启用模型自带的一次前向路径，再复用 edge 的 CPU 算子，热态耗时降至 **1.53 秒**。三种方式都选中了“编程任务”。

## 请求与测量条件

输入为“请写一个 Python 函数，合并两个有序列表。”模型在“编程任务”“数学问题”“日常聊天”中选择一项。115 token 包括状态、问题、选项和提示模板，输出为三个选项的概率，生成 token 数为 0。[完整请求](../cpu-tail-deployment/request.json)

- **硬件与软件**　16 GB RK3576，CPU 4–7，Torch 算子内 3 线程、算子间 1 线程。镜像使用 PyTorch 2.12.0+cpu 和 Transformers 5.17.0。各配置分别在限额 8 GiB 的离线容器内运行，batch 为 1。
- **模型**　使用同一份 d3.1-nano 权重，文本和视觉主干使用 BF16，决策头使用 FP32。板端核对了主权重、决策头和运行时代码的 SHA-256。[固定检查点](https://huggingface.co/vllm-sr/d3.1-nano/tree/71d06677f56e243042fe5d5f983804648a5689e8)
- **计时**　从进入 `system_one` 到返回选项概率。模型加载单独计时。NPU 首次调用包含权重转换和打包，随后重复同一输入三次，取中位数。进程峰值 RSS 包含已驻留的权重映射页。[CPU 逐次结果](../d3.1-nano-cpu-test/results.json) · [NPU 逐次结果和容器记录](results.json)

## 延迟和内存

| 模型与执行方式 | 加载后首次推理 | 热态中位数 | 进程峰值 RSS | 相对 nano 原版 CPU 加速比 |
| --- | ---: | ---: | ---: | ---: |
| d3-edge，原版纯 CPU | 20.55 秒 | 19.18 秒 | 1.15 GiB | — |
| d3.1-nano，原版纯 CPU | 108.10 秒 | 108.09 秒 | 3.28 GiB | 1.00× |
| d3.1-nano，默认前向 + NPU | 41.83 秒 | 10.87 秒 | 6.32 GiB | 9.94× |
| d3.1-nano，单问题一次前向 + NPU + edge CPU 算子 | 33.82 秒 | **1.53 秒** | 6.29 GiB | **70.68×** |

默认路径把 57-token 的状态段和 58-token 的问题段分别送过 24 层。适配器将 78 个注意力投影和 24 个 MLP 的大矩阵乘交给 RKNN，每条请求共执行 300 次 NPU 矩阵乘。模型沿用自己的 `prefix_bidir` 注意力和 FP32 决策头。[适配代码](adapter.py) · [edge 原生矩阵乘内核](../bf16-pipeline-experiment/native_mlp.cpp)

单问题配置使用官方 `D3_STATE_ONCE=0`，按相同的前缀双向注意力规则一次处理 115 token，每条请求执行 150 次 NPU 矩阵乘。edge 的 CPU 因果卷积和递推扫描各命中 18 次，小线性层命中 36 次。三次热态耗时依次为 1.554、1.529、1.516 秒。另用一次独立请求采样驱动负载，两个 NPU 核心均有活动。[板端执行与计数](run.py)

## 选项概率

| 选项 | 原版纯 CPU | 默认前向 + NPU | 单问题一次前向 + NPU | 单问题方案相对原版 CPU 的变化（百分点） |
| --- | ---: | ---: | ---: | ---: |
| 编程任务 | 99.825796% | 99.825194% | 99.830780% | +0.004984 |
| 数学问题 | 0.121012% | 0.121868% | 0.117715% | −0.003297 |
| 日常聊天 | 0.053192% | 0.052939% | 0.051505% | −0.001687 |

为单独核对矩阵乘的影响，还用独立 CPU 容器运行了一次相同的单问题前向，启用相同的三组 edge CPU 算子。与这组对照相比，单问题 NPU 方案的最大概率差为 **0.001347 个百分点**。以上延迟和概率差覆盖这一个短文本样本。[同配置 CPU 对照](cpu-control.py) · [原始概率和逐项差值](results.json)

## 复核与板端复测

在仓库根目录运行以下命令，可用已归档记录重新计算中位数、加速比及每个选项的概率差，并核对 NPU 执行计数、负载采样和容器退出状态。

```sh
python3 -B d3.1-nano-npu-test/verify.py
```

在原 RK3576 的仓库根目录可按下例复测单问题配置。脚本使用已缓存的 nano 检查点、CPU 测试目录中的 `request.json` 和 `benchmark.py`、原板 Docker 镜像、RKNN SDK 及 edge 的冻结发布目录。板端路径和镜像 ID 见[容器启动脚本](run-on-board.sh)。本次测量结束时，实验容器均已移除，原 edge 服务处于停止状态。

```bash
set -eu
nano_src=d3.1-nano-npu-test
nano_cpu=/opt/d31-nano-cpu-20261119-FCE9rf
nano_sdk=/opt/decision3-edge-matmul/sdk
nano_lab=$(mktemp -d /opt/d31-single.XXXXXX)
nano_container=${nano_lab##*/}
cp "$nano_src/adapter.py" "$nano_src/run.py" "$nano_src/cpu-control.py" "$nano_lab/"
bash "$nano_src/build.sh" bf16-pipeline-experiment "$nano_sdk" "$nano_lab/native"
bash "$nano_src/run-on-board.sh" "$nano_lab" "$nano_cpu" "$nano_container" single-question
docker inspect --format '{{.State.ExitCode}} {{.State.OOMKilled}}' "$nano_container"
docker rm -v "$nano_container"
```

本轮输出写入 `$nano_lab/out/measurements.json`。另建实验目录并将模式改为 `default` 可复测默认前向，改为 `single-question-cpu` 可生成同配置 CPU 对照 `out/cpu-control.json`。
