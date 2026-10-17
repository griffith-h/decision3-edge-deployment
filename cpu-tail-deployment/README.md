# RK3576 graph-half 服务部署（318ms 实验候选）

该版本使用完整12层和原权重，保持非因果完整注意力及DeltaNet递推逻辑。CPU融合和12个MLP RKNN子图复用，不涉及厂商库反编译或新Docker镜像构建。

**318.51ms 是已加载、暖态、batch1、115-token、单问题的 `system_one` 方法中位数。实际LAN HTTP结果见同目录 `deployment-result.json` 和 HTML 报告。它不是任意请求的延迟承诺。**

- 图输入限定 `[1,115,1024]` BF16 CPU激活，12层MLP内部使用FP16图；不支持形状/梯度/训练/权重变化/溢出等条件时回退CPU融合路径。
- 其他长度、多问题、图片仍受原模型接口约束；已验证19场景26问题及并发两个不同请求的隔离。视频支持沿用原接口，本轮未做视频端到端回归。
- 模型一次加载，图与矩阵句柄常驻；HTTP读请求可并发，模型计算通过队列在主线程串行执行，以保持Torch/OMP3线程配置，不增加吞吐承诺。
- 旧比较器忽略`confidence`。概率/score/noul最大绝对差异0.00753716、choice不变；confidence独立最大0.01002684。不能宣称所有数值字段都≤0.01。HTTP对冻结候选的比较包含confidence。

## 已验证前提

RK3576 ARM64 Debian13、16GB内存，CPU4–7，Torch/OMP3；容器Python3.12、Torch2.12 CPU、Transformers5.17；RKNN2.5.0/driver0.9.8。模型快照 `20c745653400d191ecacd99c4eafc0c5900dc15e`。

固定镜像ID `sha256:cd5c294eb741dd011ad31f8dbc6ab10a088123922122f06d2c29b7d89067d49d`。

这是一份**在已有冻结实验工作区上升级发布的说明**，不是空白RK机器的一键安装包。新机器先按仓库 `d3-edge-rk3576-install-guide.html` 基础安装，再按 `cpu-tail-experiment/REPRODUCTION.txt` 准备所有历史代码/库及12个MLP图。模型、SDK、`.so`、`.rknn`、捕获`.pt`不提交Git；源码和构建/转换步骤提交。仅克隆仓库不能直接启动。

## 1. 准备并冻结发布包

开发机器仓库根目录，静态检查和定向失败路径测试：

```sh
python3 -m py_compile cpu-tail-deployment/*.py
python3 cpu-tail-deployment/test_lifecycle.py
bash -n cpu-tail-deployment/run.sh cpu-tail-deployment/rollback.sh
rsync -az --exclude=__pycache__ cpu-tail-deployment/ root@192.168.1.164:/opt/decision3-edge-graph-deploy-tools-20261018/
```

RK宿主机执行（先核实IP、目录、镜像和旧容器名称；此部署针对本次已验证机器）：

```sh
python3 /opt/decision3-edge-graph-deploy-tools-20261018/package.py \
  --release /opt/decision3-edge-graph-20261018-v1
```

`package.py`拒绝覆盖已有发布目录；需要再次发布时新建版本目录。它先核对实验核心源码/库/12图哈希，再复制所有运行依赖到发布目录，生成`manifest.json`和`SHA256SUMS`。模型快照外部只读挂载。服务所有代码/SDK/图均只读，不依赖持续修改的实验目录；不下载、不重建镜像、不调整主机时钟/驱动。

本次冻结目录结构：

- `work/`：CPU/桥接代码、ARM库、12个MLP图。
- `deps/`：发布基座及references/combined/scan/remainder/projection/previous/attn的冻结依赖。
- `service/`：本轮HTTP入口、资源所有者、配置、预热请求。
- `run.sh`、`cutover.py`、`rollback.sh`：启动、切换、回退。

不要修改已被容器加载的`.so`或运行中的发布源码。修改配置也会使SHA校验失败；应建新版本并重新验证。

## 2. 临时8901验收，保留原8900

RK宿主机：

```sh
bash /opt/decision3-edge-graph-20261018-v1/run.sh decision3-edge-graph-staging 8901 no
```

开发机器（`http-cases.json`及冻结JSON已随源码证据提交）：

```sh
python3 cpu-tail-deployment/capture_previous.py
python3 cpu-tail-deployment/verify_http.py --port 8901 --count 20 --output staging-http.json
```

验证19场景/26问题、usage/schema、包括confidence的冻结候选输出、原0.01概率/score/noul门槛、固定形状12图命中、其他形状回退、422、模型接口、两个不同并发请求。所有延迟原样保留。任何失败先检查日志，保持旧服务；不要带着失败结果切换。

## 3. 切换原端口

RK宿主机：

```sh
python3 /opt/decision3-edge-graph-20261018-v1/cutover.py
```

切换前确认8901发布身份和发布SHA；停止并删除临时容器，将旧`decision3-edge-optimized`设为`restart=no`再停止，然后在8900启动`decision3-edge-graph`（`unless-stopped`）。若新启动/就绪失败，自动调用回退脚本恢复旧服务。停旧至新就绪有短暂中断，不宣称零停机。

重复执行会拒绝已有新容器，防止误删运行服务。中断后先检查`docker ps -a`、`cutover-result.json`和日志，再决定继续还是回退。

## 4. 回滚与恢复新版本

```sh
# RK宿主机：停止新容器并设restart=no，恢复旧三线程容器与unless-stopped策略
bash /opt/decision3-edge-graph-20261018-v1/rollback.sh
curl -fsS http://192.168.1.164:8900/health
```

需要恢复已保留的新容器时（不能同时启动两个占用8900的容器）：

```sh
docker update --restart=no decision3-edge-optimized
docker stop -t 90 decision3-edge-optimized
docker update --restart=unless-stopped decision3-edge-graph
docker start decision3-edge-graph
```

等待`/health`中release为`2026-10-18-graph-half-v1`且Docker healthy，再做HTTP验证。不要重跑`run.sh`创建同名容器，也不要删掉旧容器。旧镜像、发布文件和旧容器ID保留，只有启停状态与restart策略按切换改变。

## 5. 正式HTTP测试与日常检查

```sh
# 开发机器，保留所有100个延迟样本
python3 cpu-tail-deployment/verify_http.py --port 8900 --count 100 --output production-http.json
# 任意LAN客户端
curl -fsS http://192.168.1.164:8900/health
curl -fsS http://192.168.1.164:8900/v1/models
curl -fsS http://192.168.1.164:8900/v1/systemone \
  -H 'Content-Type: application/json' --data-binary @request.json
```

API和请求JSON格式与旧服务一致。本机原服务未配置API key，保持同样LAN访问方式；若另行配置认证，须在发布环境中显式设置并重新验证认证行为。

健康接口`deployment`包含发布标识、线程/亲和、累计图命中/回退、最近一次模型请求耗时和输入token数。计数只供诊断，不是并发请求级追踪ID。请求计数与12图命中证明每次执行了模型，不缓存答案。

```sh
docker logs --tail 80 decision3-edge-graph
docker inspect --format '{{.State.Health.Status}} {{.HostConfig.RestartPolicy.Name}}' decision3-edge-graph
```

固定CPU/内存限制为CPU4–7、8GiB；没有修改宿主机调度/驱动。启动健康宽限240秒，模型初始化时间不计入暖态延迟。

本次实际切换、回退演练、最终容器状态、SHA、19场景和100次LAN HTTP原始结果见`deployment-result.json`、`production-http.json`及HTML报告。有限样本不保证硬延迟上界。
