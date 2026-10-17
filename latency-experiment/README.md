# RK3576 独立延迟实验

本目录保存独立实验及历史证据。三线程版本的正式部署与回退见 `../optimized-deployment/README.md` 和 HTML 报告的最新部署章节；下文“原服务保持”等描述指部署之前的独立实验阶段。从零安装仍看仓库根目录的 `d3-edge-rk3576-install-guide.html`。这里的命令依赖现有板端镜像和 `/opt/decision3-edge-matmul`，不能直接当作新机器的安装脚本。

## 2026-10-17 最终结果

选择 **4 个大核（4–7）＋3 个计算线程＋`OMP_THREAD_LIMIT=3`**。在 CPU 融合算子、128 扫描分块和文本 FP32 SDPA 的基础上，减少线程争抢，为后台健康检查留出计算余量。模型权重、官方 12 层和注意力可见范围保持不变。

完整汇总见 [result-round2.json](result-round2.json)，逐次计时与校验记录保存在 [logs](logs/)。原 8900 服务的容器 ID、启动时间和示例输出均保持原样；所有临时测试容器已移除，8901 端口已关闭。

| 新进程模型调用；各预热一次 | 次数 | 中位数 / 秒 | P95 / 秒 | 最慢 / 秒 | 低于 1 秒 |
|---|---:|---:|---:|---:|---:|
| 第一轮候选实现，4 线程 | 40 | 0.989 | 1.019 | 1.829 | 37/40 |
| 第二轮最终融合算子，4 线程 | 100 | 0.824 | 0.838 | 1.397 | 98/100 |
| **第二轮最终融合算子，3 线程** | **100** | **0.865** | **0.899** | **0.987** | **100/100** |

最终 3 线程版本另测 **100 次局域网 HTTP**：中位数 **0.814 秒**、P95 **0.826 秒**、最慢 **0.879 秒**，**100/100 低于 1 秒**；无效请求返回 422。HTTP 在完整用例回归后另预热一次，与新进程模型计时分开统计。所有计时都保留最慢样本。

164 个主要算子用例及边界断言通过。最终 3 线程配置复核 7 个模型用例、14 个问题，选项全部一致；最大概率 / score 绝对差相对原 RKNN 为 **0.00363007**，相对历史优化 CPU 为 **0.00248951**（5 个用例有 CPU 参考），均低于 0.01。线程数调整会改变浮点累计顺序；3 线程相对 4 线程候选的最大差为 0.00483379。

本轮 1 秒结论只覆盖 **115 token、1 个问题、并发 1、已加载模型**。3 问题用例和图片仍需数秒，视频和高并发未测，200ms 未达到。两组各 100 次全部通过，也不能证明任意输入或长期负载下都有硬性 1 秒上限。

### 周期慢峰与健康检查

原服务每 30 秒启动一次 Python 健康检查。在中间版本 100 次 HTTP 测试中，3 次慢峰全部与该检查重叠；采样显示探针进程占用大核 5/6/7。独立执行 3 次探针，Python 客户端每次约消耗 0.9–1.0 秒 CPU；使用现有 `curl` 的 HTTP 请求本身为 2–3ms，含 Docker exec 的完整耗时约 0.18 秒。

这是后台竞争的重要证据，并非通过关闭探针获得的结果。**原探针和原服务均未修改**。最终 3 线程 HTTP 的 100 次测试期间，仍有 2 次自动健康检查在执行，全部请求保持低于 1 秒。线程余量解决了本轮测试中的慢峰。后续正式部署还可考虑用原生 curl 探针减少开销；尚未修改生产配置，也未验证替换探针后的长期表现。

```sh
# 可供后续创建容器时使用；本轮没有对现有容器应用该配置。
--health-cmd 'curl --fail --silent --show-error --max-time 5 --output /dev/null http://127.0.0.1:8900/health'
```

早期 ACTIVE/PASSIVE 线程对照中的原生卷积被保守回退条件排除，不能作为最终融合卷积的严格同条件对照。修正后，最终运行显式断言并分别记录 `norm_calls` / `conv_calls`：4 线程连续测试为 4343 / 606，3 线程连续测试加用例复核为 4644 / 648。原生卷积额外通过 10 个与向量化实现逐位相同的用例，覆盖额外透传参数和非连续输入。

目标是缩短单问题请求耗时，保留官方 12 层权重、非因果完整注意力、因果 GatedDeltaNet 和原有 CPU 回退；不使用厂家二进制逆向。验收要求为已测用例选项一致、概率 / score 最大绝对差不超过 0.01，并分别记录新进程与 HTTP 延迟。有限测试不能证明任意长度、图片、视频、并发负载的 1 秒上限。

## 实现与证据

- `cpu_attention_optimized.py`：向量化因果卷积、删去零初态及不返回的最终状态计算、仅对文本注意力使用 FP32 SDPA。第二轮扫描分块为 128；这是计算分块调整，不是裁层或改变注意力可见范围。
- `native_norm.c` / `native_norm.py`：自行实现的 ARM64 CPU 融合算子，减少 BF16 RMSNorm 和 4 抽头因果卷积的中间分配。卷积保留激活前的 BF16 舍入；门控 RMSNorm 保留原有舍入顺序。不支持的设备、类型、梯度或扫描状态路径使用原实现。
- `profile_tail_ops.py`：只在扫描范围记录算子耗时，以及进程 CPU 时间、页错误、调度切换。它本身有测量开销，不能与未插桩请求直接比较绝对速度。
- `result.json`：第一轮汇总。`result-round2.json`：第二轮汇总及选择的配置。`logs/` 保存完整样本，包括失败或变慢的试验；不删除最慢样本。
- `validate_latency.py`：原 RKNN 与候选实现成对比较，中英文、单问题、数学、聊天、图片共 7 个用例；其中 5 个另有历史优化 CPU 参考。完成后可提供临时 8901 HTTP 接口。
- `check_tail.py`：每个新进程只预热一次，再连续调用。`verify_http.py`：从 Mac 通过局域网测量临时 HTTP 接口，另有一次 HTTP 预热，且不缓存结果。

概率 / score 差不是任务准确率下降比例；这里没有完整任务数据集的准确率评测。`confidence` 不纳入该差值，与原有基准口径一致。

## 在本次已部署的 RK3576 上复现

先确认没有本目录的实验容器正在执行，再同步源码；8900 原服务可以保留。不要覆盖正在执行的脚本或已经加载的动态库。下列 IP、缓存位置、镜像和源目录属于本机环境，新机器须按自己的安装结果调整。

```sh
# Mac，在仓库根目录执行
ssh root@192.168.1.164 'docker ps --filter name=decision3-latency --format "{{.Names}}"'
rsync -av --exclude logs --exclude '*.so' --exclude __pycache__ \
  latency-experiment/ root@192.168.1.164:/opt/decision3-edge-latency/
```

板端执行，复用现有镜像，不构建镜像：

```sh
cd /opt/decision3-edge-latency
mkdir -p logs
cc -O3 -fPIC -shared -fopenmp -ffp-contract=off \
  native_norm.c -o libd3_native_norm.so -lm
bash -n run-isolated.sh

# 新进程、一次预热、100 次单问题。输出会覆盖同名记录，复测应换新名称。
bash run-isolated.sh fresh-check check_tail.py \
  --chunk-size 128 --native-norm --native-conv --count 100 \
  --output tail-reproduction.json > logs/tail-reproduction.log 2>&1

# 4 线程功能回归：164 个主要算子用例，加 7 个整模型用例，随后退出。
bash run-isolated.sh validation-check validate_latency.py \
  --chunk-size 128 --native-norm --native-conv --no-serve \
  --output validation-reproduction.json > logs/validation-reproduction.log 2>&1

# 最终选择：3 线程，保留 4 个大核；再次用 7 个用例核对上一条命令的成对参考。
D3_THREAD_LIMIT=3 bash run-isolated.sh reproduction-three check_tail.py \
  --threads 3 --chunk-size 128 --native-norm --native-conv --count 100 \
  --check-cases --case-reference /work/logs/validation-reproduction.json \
  --output tail-three-reproduction.json > logs/tail-three-reproduction.log 2>&1
```

`run-isolated.sh` 默认使用 CPU 4–7、4 个 Torch 线程、8 GiB 内存，并只读挂载生产源文件和模型缓存。最终配置同时设置 `D3_THREAD_LIMIT=3` 和 `--threads 3`，既限制 Torch，也限制原生算子的 OpenMP 并行度；CPU 仍可使用 4–7 四个大核。模型固定为 `vllm-sr/d3-edge@20c745653400d191ecacd99c4eafc0c5900dc15e`，每次加载执行完整哈希校验。测试环境为 Torch 2.12.0+cpu / Transformers 5.17.0、RKNN Runtime 2.5.0 / 驱动 0.9.8。

线程等待策略对照可在启动命令前加 `D3_OPENMP_PASSIVE=1` 或 `D3_OPENMP_ACTIVE=1`，分别对应 `PASSIVE + GOMP_SPINCOUNT=0`、`ACTIVE + GOMP_SPINCOUNT=3000000`；默认不额外设置这些变量。选用哪个配置以第二轮结果为准，不能假定主动或被动等待一定更快。公开 API 实测显示 OpenBLAS 是 `USE_OPENMP` 构建、4 线程；这种构建忽略 `OPENBLAS_NUM_THREADS`，参见 [OpenBLAS 官方说明](https://github.com/OpenMathLib/OpenBLAS#setting-the-number-of-threads-using-environment-variables)。

如需复测最终配置的 HTTP，板端在连续测试和用例复核后临时监听 8901：

```sh
D3_THREAD_LIMIT=3 D3_TEMP_HTTP=1 bash run-isolated.sh http-check check_tail.py \
  --threads 3 --chunk-size 128 --native-norm --native-conv --count 100 \
  --check-cases --case-reference /work/logs/validation-reproduction.json --serve-after \
  --output tail-http-reproduction.json > logs/tail-http-reproduction.log 2>&1
```

另开 Mac 终端，从仓库根目录执行，完成后只停止临时容器：

```sh
# 先在板端日志中确认 TEMPORARY_HTTP_READY，避免误连上一个正在退出的测试实例。
python3 latency-experiment/verify_http.py --count 100 --output http-reproduction.json
ssh root@192.168.1.164 'docker stop decision3-latency-http-check'
```

这些命令不修改生产容器配置，不启动旧 CPU 服务，不替换现有服务。模型加载和 NPU 权重缓存建立不计入推理耗时。测试请求为 115 个 token、1 个问题、并发 1；其他规模须另测。
