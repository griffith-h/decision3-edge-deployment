# 615行剩余耗时：第四阶段独立实验

这是实验复现步骤，不是线上部署脚本。先完成 [第三阶段](../long-sequence-experiment/REPRODUCTION.md) 及其上游准备，保留三个实验目录和冻结发布。沿用 RK3576、ARM64 Linux、原模型快照、原 RKNN 图和冻结推理镜像；不重新转换模型、不构建镜像、不修改厂商库。

## 隔离和资格

- 实验目录 `/opt/decision3-edge-long-tail-20261018`，容器 `decision3-long-tail-test`，无发布端口、退出自动删除。所有旧依赖和模型只读，CPU4–7、三线程、8GiB；一次只运行一个板端测试。线上 `decision3-edge-graph:8900` 保持运行。
- `baseline` 是第三阶段 both 路径；`current` 是线上 fixed115 等价路径。两者都不是纯CPU路径。
- `validate` 保留扫描计算，只将有限数与模型域检查改为自有 C/NEON 实现；`attention` 试验共享GQA转换和query分块；`mlp` 只启用已有图的并行调度；`combined` 包含三项；`validate_mlp` 保留原完整注意力，仅使用新检查和MLP调度。
- 新检查沿用第三阶段 `512<T<=2048` 和原shape/stride/参数保护。q/k/v、准备结果、衰减、beta、输出均继续检查；非有限数、非模型域、缓存/初态/最终态和未知参数仍走原回退。没有通过省略检查制造提速。
- MLP调度仅命中 `[1,615,1024]`，并要求原计划恰为 `128,128,384`。原有128图句柄连续处理两块，384图句柄独立处理最后359行（仅MLP补25行），结果按原行序收集。两个句柄分别使用公开API的core1掩码2、core0掩码1；0是AUTO，不是core0。此处是现有图调度，不是新模型或新图。
- 全部future完成后才收集结果；图执行失败时整次丢弃私有结果、恢复AUTO、禁用该层调度并重跑原bank。核心设置失败仍由外层RowBank异常保护禁用层并回退CPU。退出先等待线程池，再还原掩码，最后释放旧bank。单个Owner不支持多个请求并发调用，本轮不测并发服务。
- 其他长度、batch与图计划仍走原调度；完整模型验证范围见正式JSON，不将2048内核单测说成全模型验证到2048。

## 准备和运行

本机仓库根目录，按自己的RK机器调整地址/路径（脚本内同样需要调整）：

```bash
python3 long-tail-experiment/preserve_service.py before
ssh root@192.168.1.164 'mkdir -p /opt/decision3-edge-long-tail-20261018/logs'
rsync -a --include='*.py' --include='*.sh' --include='*.c' \
  --include='config.json' --exclude='*' \
  long-tail-experiment/ root@192.168.1.164:/opt/decision3-edge-long-tail-20261018/
```

在RK主机依次执行。主机已有GCC14/OpenMP，冻结推理容器没有GCC；新库在主机编译，容器内加载验证，其他系统须核对glibc/libgomp兼容性。不要同时运行这些命令，也不要在运行过程中更新源码或库。

```bash
cd /opt/decision3-edge-long-tail-20261018
bash build.sh > logs/build.log 2>&1
bash run.sh profile_tail.py > logs/profile.log 2>&1
bash run.sh replay.py > logs/replay.log 2>&1
bash run.sh choose_candidates.py > logs/selection.log 2>&1
bash run.sh test_candidates.py > logs/unit.log 2>&1
bash run.sh screen_tail.py > logs/screen.log 2>&1
bash run.sh formal_tail.py > logs/formal.log 2>&1
bash run.sh profile_selected.py > logs/profile-selected.log 2>&1
```

`profile_tail.py`单独捕获6层注意力、6层扫描和12层MLP的真实激活到`captured.pt`。捕获、复制和写盘不计入5次分项计时；该文件仅用于内核筛选，不提交。正式全模型的每次候选都重新执行推理，不用激活回放或答案缓存。本轮第一次profile使用的实际源码另存`logs/profile-sources/`，随后才增加候选模式；原profile数据与最终profile分开记录。

`replay.py`测试12种注意力配置和29种扫描配置（各含baseline），每种真实层输入轮转3次，结果为6层中位数之和，**不是端到端延迟**。张量allclose使用atol=.002、rtol=.008；激活最大绝对差不是概率差。`choose_candidates.py`要求组件至少快2%才替换baseline。本轮扫描保留原32行块/64列tile；完整注意力候选仍须经过全模型筛选。

`screen_tail.py`每模式4次；正式脚本先每模式6次轮转预选，在候选精度通过的前提下，combined若比保留原注意力的validate_mlp快不足2%，就优先后者。正式615行20组、115行10组，其余5/3/2组交错配对，具体见JSON。较小样本不作P95结论。

615行current重新计算并断言等于第三阶段冻结记录；其他正式场景明确复用第三阶段同一模型/路径的current完整输出。baseline和每次候选均重新推理。额外12个短请求各2次只对新算baseline，其vs_current字段是参照别名，不纳入独立current精度汇总。

精度门槛保持choice一致，概率/score/noul最大绝对差≤.01；confidence单独报告。score范围0–2，不能把score差说成概率百分点。multi-wide此前相对current的score超门槛仍保留，不因本轮增量为零说成修复。没有标注集，不宣称业务准确率无损。

`profile_selected.py`是独立于正式分布的5组配对分项计时。并行图worker的时间相互重叠，不逐个相加；MLP入口包含等待、准备、图执行与收集。所有分项按逐次统计取中位数，各列中位数不保证严格相加。

## 收尾和复核

所有自己的测试退出后，在本机按顺序执行；哈希扫描、HTTP检查和计时不要重叠：

```bash
rsync -a --exclude='__pycache__' \
  root@192.168.1.164:/opt/decision3-edge-long-tail-20261018/logs/ long-tail-experiment/logs/
rsync -a root@192.168.1.164:/opt/decision3-edge-long-tail-20261018/selected.json long-tail-experiment/
python3 long-tail-experiment/preserve_service.py after
python3 long-tail-experiment/capture_artifacts.py
python3 long-tail-experiment/validate_records.py
python3 long-tail-experiment/update_report.py
python3 long-tail-experiment/check_report.py --url http://127.0.0.1:43991/mlp-length-experiment/report.html
```

新源码/库SHA、旧依赖SHA、GCC版本和SDK核心枚举证据见`logs/artifacts.json`。SDK头文件来自已有公开开发包，不是运行库反编译。历史失败日志保留：最初单测脚本名select.py遮蔽Python标准库、随后注入别名被循环变量覆盖，均已更名修复并重跑通过；不是模型精度失败。初版C编译警告也单独保留，最终构建日志无警告。

首尾核验线上容器ID、启动时间、配置、554个发布文件SHA及5次完整HTTP输出，不用少量首尾延迟宣称性能完全相同。原服务没有部署变更；本阶段不提交推送。
