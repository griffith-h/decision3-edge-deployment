# 多长度 MLP 图实验（独立进程，不部署）

基线是 2026-10-18 已部署的 graph-half v1（原 318ms 版本）的同配置模型方法调用；不是纯 CPU。非 115 形状原来就可能使用已有 RKNN 矩阵乘和 CPU 融合路径。本实验只替换 token 独立的 MLP，不补齐整个请求、不改 attention mask、位置、扫描或权重。

## 已验证环境与依赖

- RK3576、ARM64 Debian 13，RKNN Runtime 2.5.0、驱动 0.9.8；原服务仍在 8900。
- 已存在冻结发布 `/opt/decision3-edge-graph-20261018-v1`；`run.sh` 将 work 和所有依赖以只读方式挂载。新实验目录 `/opt/decision3-edge-mlp-length-20261018` 可写，独立容器无监听端口。
- 模型快照沿用发布的原始 12 层、1024 hidden d3-edge；没有裁层、量化或修改非因果注意力。
- 容器镜像 ID 从原发布 `image-id.txt` 读取；Torch 2.12 CPU、Transformers 5.17；三线程、CPU 4–7、8GiB 上限，与现有服务相同。性能是空闲板子上隔离测试的暖机模型方法延迟，不含 HTTP，不代表并发、冷启动或视频延迟。
- 本机 x86 转换环境沿用 `attention-chunk-experiment/compiler-env`（Python3.12、Torch2.4 CPU、RKNN Toolkit2 2.5.0）和原工具镜像；Mac ARM 通过 OrbStack 的 linux/amd64 执行，未构建镜像。镜像 ID 写在 `compile.sh`。新机器先按项目安装文档准备对应环境，机器相关路径应自行调整。

## 复现步骤

先阅读 `DESIGN.txt`。以下命令在仓库根目录执行；不要向原发布目录写入代码、图或共享库，不要同时运行多个板端测试。在线服务和实验共用 NPU，测试期间其他业务请求仍会争抢资源。

1. 保存原服务身份、启动时间、配置哈希、550 个发布文件哈希以及 HTTP 输出：

   ```bash
   python3 mlp-length-experiment/preserve_service.py before
   ```

   原始随报告保存的 before/after 是本轮实测证据；重复复现时先备份，脚本会写同名文件。

2. 准备原权重捕获文件 `cpu-tail-experiment/mlp-layers.pt`。它是原模型各层 gate/up/down BF16 权重和真实 115 行激活，原始捕获方式见 `cpu-tail-experiment/capture_mlp.py`。权重、大图、ONNX 和共享库均不提交；编译时读取原权重并升为 FP32 常量，`do_quantization=False`，由公开工具生成 FP16 运算图。

   ```bash
   mkdir -p mlp-length-experiment/logs
   bash mlp-length-experiment/compile.sh > mlp-length-experiment/logs/compile-buckets.log 2>&1
   ```

   默认 128/384 两种行数，每种 12 层。每个 `logs/compile-*.json` 的 load/build/export 必须全为 0；勿仅凭 `.rknn` 文件存在判定成功。

3. 只向新目录同步源码和本轮图，原服务的 115 行图只读复用：

   ```bash
   ssh root@192.168.1.164 'mkdir -p /opt/decision3-edge-mlp-length-20261018/logs'
   rsync -a --include='*.py' --include='*.sh' --include='*.cpp' --include='*.json' --include='*.rknn' --exclude='*' \
     mlp-length-experiment/ root@192.168.1.164:/opt/decision3-edge-mlp-length-20261018/
   scp cpu-tail-experiment/mlp-layers.pt root@192.168.1.164:/opt/decision3-edge-mlp-length-20261018/
   ssh root@192.168.1.164 'bash /opt/decision3-edge-mlp-length-20261018/build_runtime.sh'
   ```

   `build_runtime.sh` 仅编译自写的公开 RKNN API 包装，读取公开 `rknn_api.h` 和原发布的 `mlp_io_codec.h`，输出新目录的 `librow_graph_runtime.so`。不修改厂商库，也不要覆盖运行中的共享库。

4. 逐项运行（每条结束再运行下一条），测试容器 `--rm` 且不发布端口：

   ```bash
   ssh root@192.168.1.164
   cd /opt/decision3-edge-mlp-length-20261018
   bash run.sh test_buckets.py > logs/bucket-unit.log 2>&1
   bash run.sh probe_buckets.py > logs/bucket-primitive.log 2>&1
   bash run.sh screen.py > logs/screen.log 2>&1
   bash run.sh screen_buckets.py > logs/bucket-screen.log 2>&1
   bash run.sh test_planned.py > logs/planned-unit.log 2>&1
   bash run.sh screen_planned.py > logs/planned-screen.log 2>&1
   bash run.sh formal_planned.py > logs/planned-formal.log 2>&1
   bash run.sh profile_planned.py > logs/planned-profile.log 2>&1
   bash run.sh screen_long.py > logs/long-screen.log 2>&1
   exit
   ```

   `test_buckets.py` 也执行行打包测试；范围拒绝须整层回退。`probe_buckets.py` 抽查第 0/5/11 层，检查补零、逐行独立、NaN/Inf/溢出拒绝与恢复，并保存 30 次原始图计时。`screen.py` 是旧 115 图拆块路线；`screen_buckets.py` 是 128/384 桶路线。`screen_planned.py` 在同一图集合上实测第 0/5/11 层的各15次暖机图调用，按中位耗时预计算行数1–2048的分块计划；这只缓存执行计划，不缓存输入/答案。正式方案的20组交替对照在 `formal_planned.py`，另有12个115-token请求各重复三轮。`formal.py` / `profile_mlp.py` 为静态桶备用脚本，本轮不默认运行。原始首次拆块实验的执行顺序和源文件另保存在 `logs/tile115-sources/`，当前脚本将较慢的图片场景移到最后，算法相同。

   原 7 场景有多问题和图片，另增加单问题 105/119/167/231/359/503 tokens 的固定输入；以实际层张量 `[B,T,1024]` 的 `B*T` 决定 MLP 行数，不能用 `usage.input_tokens` 直接选图。首次形状准备、每次暖机和正式计时分开。没有答案缓存。每个请求重跑 fixed 生成本轮参照，不使用 `http-cases.json` 中历史 `baseline` 元数据；传入模型的只有 state/questions/images/videos。

5. 收回日志并验证服务未变：

   ```bash
   rsync -a --exclude='__pycache__' root@192.168.1.164:/opt/decision3-edge-mlp-length-20261018/logs/ mlp-length-experiment/logs/
   python3 mlp-length-experiment/capture_artifacts.py
   python3 mlp-length-experiment/preserve_service.py after
   python3 mlp-length-experiment/validate_records.py
   python3 mlp-length-experiment/build_report.py
   ```

   哈希读取必须在计时进程结束后单独执行，不能与HTTP或模型计时重叠。`validate_records.py` 检查原始记录和实际命中，`build_report.py` 生成HTML。`after` 要求 ID/start/config/hostconfig/发布文件哈希不变，线上健康，五次响应与 before 完整相等，并且没有本轮测试容器残留。测试进程的失败日志应保留，不能把部分 JSON 当作整套通过。

## 判读与边界

- 概率/score/noul 最大绝对差异门槛 0.01、choice 必须相同；confidence 单独报告。`all_numeric_max_delta` 也保留，范围是 answers 中受检预测字段（排除 legend 等元数据），不能把概率门槛说成所有字段都通过。有限测试输入的一致性不能等同于业务准确率。
- `fixed`：与当前服务相同的仅 `[1,115,1024]` MLP 图路径；其他形状沿原执行路径。`tile115`：逐层展平 B*T，每115行一块，最后一块仅在 MLP 内部补零。`bucket`：115行以下复用旧图，116–128用128图，129行以上用384图分块。`planned`：使用预先实测图成本，以动态规划选不同大小的静态图组合，只有最后一块在 MLP 内补零，115行沿原路径。这些均为静态图复用原型，不是真正任意动态形状 RKNN 图。
- 现有底层矩阵适配器的 eligibility 上限为512行，这是当前软件配置边界，不是硬件上限的结论。`screen_long.py` 补测原始输入511/512/513/615行：各模式先暖机一次、交替计时两次，首次形状开销单列；样本量较小，不当作P95或SLA。
- 当前代码最多2048个 MLP 行；超限、非 BF16/CPU/三维输入、训练/梯度、权重或模块变化等走原路径，无截断。运行错误会禁用故障层，范围失败仅整层回退。正常测试必须检查 errors/range_fallback/fallback，而非只看图命中次数。
- 已验证单请求、文本和一张图片；未验证视频、多请求吞吐或在其他 RK 芯片上的行为。更换权重/发布目录/SDK需重新转换和验收。
- 本轮没有替换服务。新原型即使更快，也仍需按实际请求长度和输入分布决定上线策略。

公开 SDK 同时提供动态形状 API；本轮没有测试原生动态 profile 的编译/性能，不能据此断言 RKNN 不支持动态。本原型用静态图复用处理不同长度，速度和精度以各输入的实际记录为准。
