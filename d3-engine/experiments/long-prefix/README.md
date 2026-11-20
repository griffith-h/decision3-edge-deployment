# d3.1-nano 长前缀原型（RK3576）

在同一块 16 GB RK3576 的同次启动中，八核正式版处理 55 题发票的热态中位数为 679.15 秒。使用本目录的实验快照，CPU 与 NPU 同时分担长查询后为 619.63 秒，合并投影后为 607.07 秒，调整分批后为 601.03 秒。最后按注意力形状分工，两轮为 598.32 / 599.21 秒。各步保存了相同输入、每个选项概率、板端环境和内存峰值。[完整对照表与精度](../../EIGHT-CORE.md) · [设备状态](../../../WORK_STATUS.md)

| 原型 | 测量对象 | 现有证据 |
| --- | --- | --- |
| CPU 与 NPU 分担查询行 | `--aggressive --prefix-backend hybrid` 对受支持且查询数 ≥64 的长段，将约 3/8 的查询行交给 ARM，其余行交给两个 NPU 核。两边共享完整 K/V，结束时按原次序拼回。 | 55 题发票两次为 619.41 / 619.86 秒，峰值 RSS / cgroup 为 8.44 / 5.72 GiB。同次八核对照的 171 个选项最大差 0.8526 个百分点。[逐选项核对](../../results/long-prefix/hybrid-vs-eight.json) |
| 合并投影 | `--fuse-projections` 把同一输入上的相邻投影合成一次矩阵乘，再拆回各自输出。包含注意力 Q/K/V 和递推层 QKV/Z。 | 用真实 RKNN 执行，订单和家电每次分别合并 48 和 144 批投影。订单热态 14.69 秒，家电两轮 38.59 / 39.23 秒。相对原版纯 CPU，22 个判断一致，55/57 个选项在 1 个百分点内。[两条完整请求](../../results/long-prefix/fused-short.json) |
| Mali FP32 矩阵乘 | 在已驻留 Mali-G52 的真实发票 Q/K 上，用 CLBlast 分块计算并同步 QKᵀ。每轮校验 1,224 个分数。 | 两轮热态 2.533 / 2.531 秒，包含块复制和抽查为 3.709 / 3.714 秒。同形状的 CPU 与 RKNPU 完整注意力为 0.601 / 0.619 秒。[GPU 原始计时](../../results/long-prefix/mali.jsonl) |

## 真实张量重放

断线前完成的 16 条重放记录已从原板取回并校验 SHA-256。下表使用同一层、同一组真实 Q/K/V，每格为两次运行的秒数。问题段首次调用包括 RKNN 初始化。[逐次时间、核心用量与误差](../../results/long-prefix/query-split-probe.json) · [张量形状与哈希](../../results/eight-core/prefix-probe.json)

| ARM 分担的查询行 | 状态段，8,902 个位置 | 问题段，4 题各 130 个位置 |
| ---: | ---: | ---: |
| 0 | 11.666 / 10.995 | 2.128 / 0.738 |
| 1/4 | 8.988 / 9.091 | 0.668 / 0.707 |
| 3/8 | 8.586 / 8.147 | 0.601 / 0.619 |
| 1/2 | 7.811 / 7.835 | 0.639 / 0.612 |

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

`batch-padding.patch` 是基于这个实验快照的独立改动。它在检查点已经排好的题序内选择分批边界，使各批「题数 × 最长题长」之和最小。批数和每批最多 4 题的设置保持相同。复现该组时，在上面克隆的副本中追加执行 `git -C "$scratch" apply "$study/batch-padding.patch"`。在发票的 26,092 个真实问题位置上，15 批的实际计算位置由 27,771 个降到 27,288 个。[分批实现与行序测试](batch-padding.patch)

在该快照上再执行 `git -C "$scratch" apply "$study/square-split.patch"`，可复现按长注意力形状分工的实验。单序列且查询行数与键数相等时，CPU 分担 1/2，其余符合分行条件的段继续用 3/8。张量重放中，8,902 个位置的状态段按 1/2 分配用时 7.811 / 7.835 秒。问题段按 3/8 分配为 0.601 / 0.619 秒。[分工实现与形状测试](square-split.patch) · [张量原始计时](../../results/long-prefix/query-split-probe.json)

在已装好引擎对应 Torch 和 Transformers 的 Python 环境中，可本地验证分行与回退。

```bash
PYTHONPATH="$scratch/d3-engine:$scratch/d3.1-nano-npu-test:$scratch/bf16-pipeline-experiment" \
    python -B -m unittest test_engine.EngineTests.test_concurrent_query_parts_keep_row_order_and_propagate_range_fallback
```

## 板端复现所需输入

整单对照沿用正式基准的同一份发票、权重、线程数与预热方式。将补丁版引擎放入独立实验目录，按正式基准准备 `$lab/src/order.json` 和唯一一份 `typesafe-invoice-*.json`。以下两次运行使用相同的输入和核心分配，各自保存模型进程 RSS、容器峰值、计数和完整响应。[引擎环境与样本](../../README.md) · [正式版四核与八核配对命令](../../EIGHT-CORE.md#复测)

```bash
invoice=( "$lab"/src/typesafe-invoice-*.json )
cases=( /bench/order.json "/bench/$(basename "${invoice[0]}")" )
export D3_CPUS=0-7 D3_THREADS=8 D3_BLAS_THREADS=1 D3_OMP_WAIT_POLICY=PASSIVE
bash "$lab/src/run-board.sh" "$lab" "$checkpoint" "$run_id-hybrid" rknn \
    /out/hybrid.json "${cases[@]}" --aggressive --prefix-backend hybrid --repeats 2
bash "$lab/src/run-board.sh" "$lab" "$checkpoint" "$run_id-combined" rknn \
    /out/combined.json "${cases[@]}" --aggressive --prefix-backend hybrid --fuse-projections --repeats 2
```

张量重放需要之前使用 `bench.py --capture-prefix` 保存的 `questions.pt` 和 `state.pt`。在准备好实验快照和 RKNN 原生库的基准容器内运行 `python -B probe_overlap.py /captures /out/overlap.json --threads 8`，并将补丁后的引擎与 native 库目录放入 `PYTHONPATH`。每条记录含墙钟时间、平均 CPU 核心数，以及相对 ARM 参照的输出和 log-sum-exp 最大差。[分行重放脚本](probe_overlap.py)

Mali 探针需要板上已有的 CLBlast/OpenCL 编译依赖、edge 实验解包的 GPU 运行库和正式基准镜像。手工入口为 `bash run-mali-probe.sh <补丁后的d3-engine目录> <采集目录> <GPU运行库根目录> <已存在的输出目录> <本地镜像ID>`。两步分别导出 FP32 输入和测量驻留矩阵乘，使用独立无网络容器；三轮均完成后才产生 `mali.jsonl`。张量、模型、运行库和未完成计时均不随源码分发。[Mali 探针](probe_mali.cpp) · [隔离运行脚本](run-mali-probe.sh) · [固定镜像与权重](../../results/eight-core/hardware.json)

命令末尾加 `--tiled` 会把相同发票 QKᵀ 的全部分数拆成最多 128 行查询、1,024 个键的块，在 Mali 上逐块计算并同步。两头共用的输出缓冲区最多 1 MiB。每块两头各抽查四个位置，与原输入的双精度点积对照。`seconds` 累计矩阵乘提交与等待时间，`wall_with_copies_and_checks_s` 还包括设备内复制和抽查。两项分别记录，有助于判断单是 QKᵀ 的成本是否已超过其他后端的完整注意力。
