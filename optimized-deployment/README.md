# RK3576 三线程版本的部署与回退

本目录部署已验证的 `2026-10-17-three-thread-v1`，沿用 CPU + RKNN MatMul，不整模型转换、不修改权重或裁层。CPU 4–7，Torch/OpenMP 计算线程 3，OpenMP 上限 3，扫描块 128，文本 FP32 SDPA，CPU 原生 BF16 RMSNorm 和四抽头因果卷积。完整注意力仍非因果，GatedDeltaNet 扫描仍因果。

## 前提

新机器先按根目录 `d3-edge-rk3576-install-guide.html` 完成基础安装、固定模型下载、SDK/桥接库构建及原 RKNN 服务核验。本节是其后的优化升级步骤，不包含镜像、模型或 SDK 二进制分发。已验证 RK3576 / ARM64 Linux、RKNPU 0.9.8、RKNN Runtime 2.5.0、Torch 2.12.0+cpu、Transformers 5.17.0；其他版本需要重新验证。

已有 `decision3-edge:arm64-cpu` 镜像，原容器 `decision3-edge-matmul`、目录 `/opt/decision3-edge-matmul`，以及原模型缓存。以下在板端 root shell 执行；先将本仓库检出到 `/opt/decision3-edge-deployment`。本次部署复用镜像，不重新构建。

## 从零安装指南与本机目录的衔接

安装指南生成的是 `d3-edge-rknn:1.0` 镜像、`d3-edge-rknn` 容器、`$D3_HOME/data/models/d3-edge`。它们与原测试机路径不同。若使用该指南，先在已完成基线验收的板端执行以下兼容准备，再执行后面的升级步骤。此衔接路径经过脚本/目录静态审查，未在空白机器完整重跑；接收方仍需真实验收。

```bash
# D3_HOME 必须指向安装指南生成的 d3-edge-rknn 工作目录。
export D3_HOME=/自己的路径/d3-edge-rknn
sudo apt-get update
sudo apt-get install -y build-essential
# 只是给现有镜像加一个兼容标签，不构建镜像。
docker tag d3-edge-rknn:1.0 decision3-edge:arm64-cpu
# 以下名称若已存在先核对，不能覆盖其他部署。
docker rename d3-edge-rknn decision3-edge-matmul
test ! -e /opt/decision3-edge-matmul
mkdir -p /opt/decision3-edge-matmul/sdk
docker create --name d3-upgrade-export d3-edge-rknn:1.0
for file in cpu_optimized_server.py rknn_linear.py libmatmul_bridge.so request.json; do
  docker cp "d3-upgrade-export:/work/$file" /opt/decision3-edge-matmul/
done
docker cp d3-upgrade-export:/work/sdk/librknnrt.so /opt/decision3-edge-matmul/sdk/
docker cp d3-upgrade-export:/tmp/sdk-fetch/sdk-manifest.json /opt/decision3-edge-matmul/
docker rm -v d3-upgrade-export
# 用挂载根目录内部的相对链接兼容固定 snapshot 路径，不复制权重。
mkdir -p "$D3_HOME/data/models/models--vllm-sr--d3-edge/snapshots"
ln -s ../../d3-edge "$D3_HOME/data/models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e"
export D3_MODEL_ROOT="$D3_HOME/data/models"
# 之后在同一个 shell 运行 run.sh；此变量要传到首次创建的容器命令。
```

新镜像必须有 `curl`，用 `docker run --rm --entrypoint curl decision3-edge:arm64-cpu --version` 核验；若没有，接收方需在自己的镜像安装 curl 后再创建服务。旧容器是从零指南创建并改名的原容器，不重新创建，以便回退。

## 准备独立目录

```bash
set -euo pipefail
cd /opt/decision3-edge-deployment
release=/opt/decision3-edge-optimized-20261017-v1
# 新目录：若已存在先检查，不覆盖正在运行的版本。
test ! -e "$release"
mkdir -p "$release/base/sdk"
cp optimized-deployment/{server.py,run.sh,rollback.sh} "$release/"
cp latency-experiment/{cpu_attention_optimized.py,native_norm.py,native_norm.c} "$release/"
cp /opt/decision3-edge-matmul/{cpu_optimized_server.py,rknn_linear.py,libmatmul_bridge.so,request.json,sdk-manifest.json} "$release/base/"
cp /opt/decision3-edge-matmul/sdk/librknnrt.so "$release/base/sdk/"
docker image inspect decision3-edge:arm64-cpu --format '{{.Id}}' > "$release/image-id.txt"
docker inspect decision3-edge-matmul > "$release/original-container.json"
cd "$release"
cc -Wall -Wextra -O3 -fPIC -shared -fopenmp -ffp-contract=off native_norm.c -o libd3_native_norm.so -lm
sha256sum server.py cpu_attention_optimized.py native_norm.py native_norm.c libd3_native_norm.so base/*.py base/*.so base/sdk/*.so > SHA256SUMS
```

需板端 C 编译器和 OpenMP（Debian 的 `build-essential`）；`.so` 必须在 ARM64 板端编译，不能复制 Mac 产物。此步骤复制原 SDK 与桥接库到版本目录，不改原目录。`image-id.txt` 固定实际使用的镜像 ID，保留该镜像以便重启。不要覆盖运行中只读挂载的源文件。

## 8901 验证后切换 8900

以下 IP 必须改为接收方机器自己的局域网 IP。`D3_BIND_IP` 仅影响创建容器，不会自动修改既有容器。可用 `D3_MODEL_ROOT` 指定不同缓存根目录，内部 snapshot 层级保持不变。

```bash
cd /opt/decision3-edge-optimized-20261017-v1
export D3_BIND_IP=192.168.1.164
bash run.sh decision3-edge-optimized-staging 8901 no
# 加载及完整哈希校验需要时间；重复检查直到 status=ready。
curl -f http://192.168.1.164:8901/health
# deployment.release=2026-10-17-three-thread-v1，torch_threads=3，
# OMP_THREAD_LIMIT="3"，affinity=[4,5,6,7]，norm_calls/conv_calls > 0。
# 将报告中的调用示例端口改为 8901，核验实际分类及概率。

# 暂停切换前，确认上面的接口与输出均通过。
docker stop -t 60 decision3-edge-optimized-staging
# 日志应含 SERVICE_STOPPED_CLEANLY，退出码 0；没有 OOM。
docker inspect decision3-edge-optimized-staging --format '{{.State.ExitCode}} {{.State.OOMKilled}}'
docker rm -v decision3-edge-optimized-staging
# 保留原容器；禁用其自动重启，避免与优化版争用 8900。
docker update --restart=no decision3-edge-matmul
docker stop -t 60 decision3-edge-matmul
# 如果创建/加载新服务失败，执行下一节的回退。
bash run.sh decision3-edge-optimized 8900 unless-stopped
curl -f http://192.168.1.164:8900/health
# 等待 ready 后使用真实请求核验；不要仅凭 docker run 成功判断就绪。
docker inspect decision3-edge-optimized --format '{{.State.Status}} {{.State.Health.Status}}'
```

新容器 `unless-stopped`，原容器 `no`，CPU 8899 备用容器保持停止。新版本 curl 探针每 30 秒一次；`--init` 和 SIGTERM 处理允许当前推理结束后释放 NPU 缓冲区。切换期间存在短暂不可用窗口。端口映射不能热改，因此验证容器结束后创建正式容器。

## 一键回退及重新启用优化版

```bash
bash /opt/decision3-edge-optimized-20261017-v1/rollback.sh
# 预期：优化版停止，原 decision3-edge-matmul 在 8900 ready。
# 原版保留自己的 Python 探针、4 线程及 restart=no；回退后重启需手动 start。
# 回退脚本 IP 如有变化也要先修改。

# 再次启用同一个优化版容器，不重建、不删除原版：
docker stop -t 60 decision3-edge-matmul
docker start decision3-edge-optimized
curl -f http://192.168.1.164:8900/health
```

停止优化容器后 `unless-stopped` 会保持其手动停止状态。原容器、源目录、模型缓存和镜像均保留，不执行 prune。报告记录一次实际回退及再次启用的核验。

## 复现测量

在能访问板端的电脑上，仓库根目录运行（Python 3，仅标准库）：

```bash
python3 latency-experiment/verify_http.py --base http://192.168.1.164:8900 \
  --count 100 --output my-production-http.json
```

脚本先验证 ready，再预热 1 次，计时 100 个串行真实 HTTP 请求，校验选择不变、相对保存的 CPU 参考概率 / score 最大绝对差 ≤0.01，另验证无效请求 422。输出位于 `latency-experiment/logs/my-production-http.json`，包含每次耗时、P95、最慢值，不缓存结果。默认输入是根目录 `request.json` 的 115 token 单问题；图片、多个问题及并发不属于 1 秒结论。

7 个模型用例/14 个问题回归及 164 个算子用例的复现见 `latency-experiment/README.md`。它们是计算路径验证，不是完整任务数据集准确率评测。最终部署记录 `deployment-result.json`，线上 100 次 HTTP 记录见报告内置原始数据。
