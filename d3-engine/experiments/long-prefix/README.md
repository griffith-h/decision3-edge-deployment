# d3.1-nano 长前缀原型（RK3576）

八核正式基准采用长段双核 RKNPU、短段 ARM，55 题发票热态中位数为 679.69 秒。本目录保存同一轮开发留下的三个原型，缺少三者各自的整单耗时、峰值内存与逐选项概率。Mali 诊断期间板端连接中断，尚未取回完整 GPU 计时。[正式基准与原始记录](../../EIGHT-CORE.md) · [设备状态](../../../WORK_STATUS.md)

| 原型 | 测量对象 | 现有证据 |
| --- | --- | --- |
| CPU 与 NPU 分担查询行 | `--aggressive --prefix-backend hybrid` 对受支持且查询数 ≥64 的段，将约 3/8 的查询行交给 ARM，其余行交给两个 NPU 核。两边共享完整 K/V，结束时按原次序拼回。 | 主机测试覆盖行序与 NPU 返回越界时的回退。张量重放探针可比较 0、1/4、3/8、1/2 四种分配。 |
| 合并投影 | `--aggressive --prefix-backend rknn --fuse-projections` 把同一输入上的相邻投影合成一次矩阵乘，再拆回各自输出。包含注意力 Q/K/V 和递推层 QKV/Z。 | 源码保留权重变动与不支持输入的回退；真实 RKNN 合并、内存成本与端到端收益待板测。 |
| Mali FP32 矩阵乘 | 在已驻留 GPU 的真实发票 Q/K 上，用 CLBlast 计算并同步 QKᵀ，每轮对两头各抽查四个分数。 | 探针仅计这一项乘法；完整注意力还需要 softmax、乘 V 和传输。尚无可报告的 GPU 用时。 |

## 在本机还原实验代码

`candidate.patch` 保存七个 Python 文件相对提交 `4606a73fe9a4c2ef3df728099ab4baa61ca3f1ed` 的原始差异。以下命令从仓库根目录创建独立临时副本，只使用本地 Git 对象。实验快照里的 `--aggressive` 单独使用会选 ARM，比较时须显式传入表中的后端。

```bash
study="$PWD/d3-engine/experiments/long-prefix"
scratch=$(mktemp -d)
git clone -q --local --no-checkout . "$scratch"
git -C "$scratch" checkout -q --detach 4606a73fe9a4c2ef3df728099ab4baa61ca3f1ed
git -C "$scratch" apply --check --index "$study/candidate.patch"
git -C "$scratch" apply "$study/candidate.patch"
```

在已装好引擎对应 Torch 和 Transformers 的 Python 环境中，可本地验证分行与回退。

```bash
PYTHONPATH="$scratch/d3-engine:$scratch/d3.1-nano-npu-test:$scratch/bf16-pipeline-experiment" \
    python -B -m unittest test_engine.EngineTests.test_concurrent_query_parts_keep_row_order_and_propagate_range_fallback
```

## 板端复现所需输入

整单对照沿用正式基准的同一份发票、权重、线程数与预热方式。一次只改查询分工或合并投影；分别保存热态耗时、RSS、容器峰值和完整响应，使用同一参照比较所有选项。已有四核与八核的配对命令可作为起点。[引擎环境与样本](../../README.md) · [配对命令](../../EIGHT-CORE.md#复测)

张量重放需要之前使用 `bench.py --capture-prefix` 保存的 `questions.pt` 和 `state.pt`。在准备好实验快照和 RKNN 原生库的基准容器内运行 `python -B probe_overlap.py /captures /out/overlap.json --threads 8`，并将补丁后的引擎与 native 库目录放入 `PYTHONPATH`。每条记录含墙钟时间、平均 CPU 核心数，以及相对 ARM 参照的输出和 log-sum-exp 最大差。[分行重放脚本](probe_overlap.py)

Mali 探针需要板上已有的 CLBlast/OpenCL 编译依赖、edge 实验解包的 GPU 运行库和正式基准镜像。手工入口为 `bash run-mali-probe.sh <补丁后的d3-engine目录> <采集目录> <GPU运行库根目录> <已存在的输出目录> <本地镜像ID>`。两步分别导出 FP32 输入和测量驻留矩阵乘，使用独立无网络容器；三轮均完成后才产生 `mali.jsonl`。张量、模型、运行库和未完成计时均不随源码分发。[Mali 探针](probe_mali.cpp) · [隔离运行脚本](run-mali-probe.sh) · [固定镜像与权重](../../results/eight-core/hardware.json)
