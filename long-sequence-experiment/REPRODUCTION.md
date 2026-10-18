# 长序列注意力与扫描：独立实验复现

沿用已验证的 RK3576、ARM64 Linux、冻结 graph-half v1 镜像及 SDK。先完成上两阶段 [MLP 图](../mlp-length-experiment/REPRODUCTION.md) 和 [其他投影分块](../row-projection-experiment/REPRODUCTION.md) 的准备；本阶段不转换模型或构建镜像。

本阶段改动的是我们自己编写的 CPU C 内核及 Python 资格检查。两份原 C 源码的 SHA 与冻结发布中的对应文件一致；主体计算不变，扩大长度检查并增加资源边界。不是 RKLLM 支持，也没有修改或反编译厂商库。

## 隔离与适用范围

- 当前服务 `decision3-edge-graph:8900` 保持运行，发布目录 `/opt/decision3-edge-graph-20261018-v1` 全部只读挂载。
- 前两阶段目录分别只读挂载为 `/mlp` 和 `/row`。新目录 `/opt/decision3-edge-long-sequence-20261018` 挂载为 `/experiment`，只在此生成新 `.so` 和日志。
- 测试容器 `decision3-long-sequence-test` 无发布端口、`--rm`、CPU4–7、三线程、内存上限8GiB。板端一次只运行一个测试进程。
- 新路径只处理 `512<T<=2048`。512及以下仍委托原路径；2049以上、缓存/初始状态/最终状态、因果注意力、未知参数或不支持的签名仍回退。这个上限是本轮有界适配范围，不是硬件或模型的长度上限。
- 注意力：CPU/BF16，最多8个batch、64个query head、batch×head最多128、head dim最多256，保留完整 K/V、GQA 和原布尔 padding mask。所有 query 都能访问未屏蔽的未来 key。
- 扫描：CPU/BF16 q/k/v、BF16或FP32 g/beta，最多8个batch、batch×head最多128、K/V维度不超过128且为4的倍数；零初态、无需最终态。保持原32行块/64列 tile，在一次完整调用内连续传递状态；不把不同512行段当作新零状态请求。
- 同时仍受上阶段线性投影 `B*T<=2048` 的独立扩展保护约束。长序列的batch大小也要结合这个边界判断，不能只看T。
- 原生失败时丢弃整个私有输出，完整调用原实现并禁用该故障内核扩展；没有局部输出拼接到旧输出。有限数、扫描模型域、shape/stride/mask/参数保护见 `long_kernels.py`。

## 准备与编译

本机仓库根目录执行；地址和路径按对方自己的机器调整：

```bash
python3 long-sequence-experiment/preserve_service.py before
ssh root@192.168.1.164 'mkdir -p /opt/decision3-edge-long-sequence-20261018/logs'
rsync -a --include='*.py' --include='*.sh' --include='*.c' \
  --include='config.json' --include='natural-request.txt' --exclude='*' \
  long-sequence-experiment/ root@192.168.1.164:/opt/decision3-edge-long-sequence-20261018/
```

在 RK 主机编译我们的 C 源码，随后在冻结容器内运行。主机已有 GCC14 和 OpenMP；本阶段没有安装软件。冻结推理镜像没有 GCC，首次尝试在容器内编译因此失败，日志保存在 `logs/build-container-missing-gcc.log`；改用主机编译后在容器验证加载和计算。

```bash
cd /opt/decision3-edge-long-sequence-20261018
bash build.sh > logs/build.log 2>&1
bash run.sh test_kernels.py > logs/unit.log 2>&1
bash run.sh screen.py > logs/screen.log 2>&1
# 审阅单项筛选结果后，正式组合路径：
bash run.sh formal.py --candidate both > logs/formal.log 2>&1
```

其他系统重新编译时需确认主机 glibc/libgomp 与容器兼容。若不兼容，在匹配容器运行环境的工具链中编译；不能把本机成功加载推断成任意系统兼容。

## 模式、计时与精度

- `current`：线上 fixed115 等价路径，新长内核关闭。
- `baseline`：上一阶段 row-fused 快路径，新增长内核关闭。它是本阶段速度基线；不能与纯 CPU 或多问题原 MLP 保护版混淆。
- `attention` / `scan` / `both`：在 baseline 上仅启用相应的长序列扩展。
- `screen.py`：每输入每模式3次暖机后的轮转计时。513/615的 current 参照来自上轮同一冻结模型记录，1024只比较新算的 baseline；正式实验重新计算每个输入的 current。
- `formal.py`：115行20组配对；513/615/1024分别5次 baseline 控制和20次候选；长英文请求与长批次分别3次 baseline 和10次候选；512行及原多问题短序列各3次配对，图片各2次。均交错分段，先暖机；九个正式场景分别重新计算 current 与 baseline 作为评估参照，不缓存候选答案。额外12个115-token请求各2次只核对新算的baseline，原始记录中的vs_current复用了该参照，并非独立current测量，不计入报告的current误差汇总。
- 正式每次保存相对 baseline 和 current 的概率/score/noul、choice、confidence差异。原绝对0.01门槛不放宽，confidence独立列出。已有 MLP 图组合的偏差与本轮新内核引入的偏差要分开，不因 native 命中就宣称精度通过。
- `natural-request.txt` 是手写英文软件需求探针；alpha输入是重复词长度探针。长批次是两种分类问题的padding探针，均不代表真实业务分布或标注准确率。
- 615行另有每模式3次带计时包装的组件分解，保存在 `formal.json` 的 `profile`，不混入正式分布。
- 正式计时后补充了注意力 `scaling=None/NaN` 的原样回退守卫，并重跑聚焦单测。标准模型使用的有限浮点scale、C计算与两个二进制均未改变；没有为这个非活动分支重跑整套模型计时。计时源码SHA在 `logs/measured-artifacts.json`，原适配器与单测保存在 `logs/pre-scale-guard/`，最终差异见 `logs/scale-guard.diff`；最终运行源SHA在 `logs/artifacts.json`。
- 当前参照的首次准备耗时单列，不用于速度分布。记录句柄数复用、环境状态及进程RSS高水位；RSS不代表NPU驱动总内存。模型方法延迟不含HTTP/排队，不是多请求SLA。

## 收尾与报告

等所有计时进程退出后，本机顺序运行，哈希扫描不与计时或HTTP测量重叠：

```bash
rsync -a --exclude='__pycache__' root@192.168.1.164:/opt/decision3-edge-long-sequence-20261018/logs/ long-sequence-experiment/logs/
python3 long-sequence-experiment/preserve_service.py after
python3 long-sequence-experiment/capture_artifacts.py
python3 long-sequence-experiment/validate_records.py
python3 long-sequence-experiment/update_report.py
```

现有报告更新到 `../mlp-length-experiment/report.html#long-sequence`，前两阶段数据和失败路线继续保留。本阶段没有部署、提交或推送。
