# 第五阶段：融合与流水（验收完成，1秒目标未达成）

目标：B1/T615暖态方法P50/P95均低于1秒，维持原数值门槛和完整模型语义。此目录为独立实验，不是部署方案。线上冻结版本及四个旧实验只读。

板端 `/opt/decision3-edge-fusion-pipeline-20261018`，无端口容器 `decision3-fusion-pipeline-test`；沿用CPU4–7、3线程、8GiB、固定镜像。一次只运行一个板端实验。上传、源码/库重建必须等待本轮进程退出；哈希核验/HTTP保护与计时分开。

`preserve_service.py before/after`保存线上身份、配置、554个发布文件SHA和固定HTTP输出。原始JSON、所有失败日志、运行源码快照在logs；未生成完整记录的运行不能算通过。

当前路径：

- `profile_initial.py`：复核validate_mlp，并在独立回放中分解自有注意力C算子的阶段。worker_seconds是工作线程耗时之和，不能当作串行墙钟时间相加。
- `profile_projection.py`：使用冻结自有profile桥接，SDK区间包含提交、驱动及等待，不是纯NPU计算；各worker时间重叠。
- `dynamic_matmul.cpp`、`native_attention.inc`：公开MatMul的FP32输出、CPU FP32 softmax、动态B更新。与历史FP16完整注意力图不同。B写入后显式sync并rebind。NPU仅为对齐内部补零，全部真实K/V与mask保留。
- `diagnose_dynamic_b.py`、`probe_npu_attention.py`：小矩阵精度/重复性和六真实层回放，仅组件证据。
- `screen_npu.py`：新算current、validate_mlp，交错筛选Python桥接、融合桥接以及低半精度残差补偿模式。全模型输出逐份保存。
- `screen_wide.py`：同一扩展shape银行内比较512+103与615一次提交，原有权重和舍入不变。
- `screen_staging.py`：验证原始C缓冲整体复制后转换是否减少读开销，原始位模式不变。
- `probe_mlp640.py`：原权重640行图的公开core mask0/1/2/3组件回放。mask3只指图接口；不推广为MatMul支持组合掩码。
- `fused_scan.c`：保持32行块/64列tile，将Q/K的同右矩阵GEMM合并，复用打包数据；状态仍逐块完整传递。与旧参数扫描、直接递推不同。

本地构图使用原固定x86工具镜像、公开Toolkit2和原BF16权重提升FP32常量，不量化、不改权重。`bash compile.sh --rows 640 --layers ...`输出本目录。运行/模型/捕获/二进制按旧忽略规则不入Git。

最终615行P50 1745.16ms、P95 1776.55ms；原34例数值门槛通过，但1秒目标未达到。详见统一报告第五阶段、logs/summary.json及VERIFY.txt。

## 已落实的故障边界

`test_components.py`覆盖注意力未来key、Q/K/V非有限数、全遮罩、重复调用，以及原生故障后等待全部worker；卷积候选原BF16边界和旁路。`test_full_mlp.py`覆盖640图私有补零、失败整层旧bank回退、故障层禁用、构造失败清理和重复关闭。`test_precise.py`覆盖第二块失败后完整CPU MLP重算、等待双worker、运行故障禁用后持续CPU回退（不能重新进入已知失败的旧图）及长序列旁路。`test_split.py`覆盖输出列顺序、范围/运行错误的全部等待、权重版本保护及恢复。各测试是否实际完成以日志JSON为准。

## 新增路线的证据边界

扫描仅4线程的实验使用 `run4.sh`：Torch/OMP_NUM_THREADS仍3，OMP_THREAD_LIMIT为4，只有自有扫描函数显式请求4线程；CPU仍4–7。旧29项扫描只含2/3线程，R24为115行全局线程数试验，不能代替本轮长序列混合执行对照。32行块/64列tile与完整FP32跨块状态不变。

`neon_scan.c`进一步尝试用自有4x8 FP32矩阵核处理预测和状态更新，不能把它与四线程原BLAS路线混为一项。`profile_scan.py`的分项为工作线程时间之和，不是端到端时间。

`test_grouped.py`/`screen_grouped.py`尝试将共享K/V的两个独立query head合为MatMul行，M=1230；每行仍独立对完整615个key归一化。全模型收益不足2%时保持单head提交，不因回放等价就采用。

最终候选由 `execution.json`记录，原有 `selected.json`继续是第四阶段组件参数，不得混用。已知形状的旧MLP计划从第四阶段正式JSON冻结，避免重新校准改变基线；615仍128+128+384。最终参照由 `make_references.py`在原始runner/current、原3线程环境单独重算；候选和baseline每次都重新推理。

## 扩展精度对照与最终验收流程

`make_references.py`仅使用原`runner.Owner/current`，在`run.sh`原三线程环境新算34个参照，已有九例须与旧冻结记录完全相等。新增12例不同内容的615 token和12例115 token，另有一例短多batch。保存参照仅用于验收，推理路径不读取答案。第一轮组合回归发现matched615-1/9超原概率门槛，旧baseline亦失败，见`regression-research-stack.json`及`equivalence-before-precision.json`，不得以增量小为由忽略。

`precise_mlp.py`的615资格恢复原BF16 gate/up、SiLU和乘法边界，原权重及顺序不变；512/640行native提交仅内部补零。全部块输出保持私有，任何块失败整层CPU重算，运行失败后该层持续CPU。`screen_precision615.py`检查13个615内容，`ablate_precision_final.py`在同一资源银行内交错筛选提交宽度和末层路径。最终`execution.json`记录选定资格；原有`selected.json`只记录第四轮参数。

`last_readout.py`只包装原D3.logits私有消费者，不替换公开encoder。首版组件allclose(.002,.008)在隐藏最大差0.0078125时失败，保存为`readout-unit-initial.json`和初始测试源码。复核保留该allclose结果，另明确检查隐藏最大绝对差≤0.01并要求原完整输出门槛；隐藏界限是组件诊断，不是用户的概率/score门槛。不同形状CPU/NPU MLP舍入可产生该小幅差异，不能宣称逐位等价。原输出choice、概率/score/noul≤0.01标准从未改变。

正式计时三轮独立容器，每模式三次暖机、30组交错配对，nearest-rank P95，所有慢样本保留。`formal_selected.py`检查615 usage、实际优化调用数、全部数值门槛、无故障/范围回退、暖态engine句柄及FD稳定、关闭后句柄0。`short_selected.py`另做12组115 token对照，不用于615目标。

最终板端操作须等前一进程退出，不能在运行中覆盖源码/库。`build-selected.sh`只构建采用路径；所有控制都持有同样的资源银行：

```bash
# 已准备冻结依赖的RK主机，新实验目录内；顺序执行，无端口
bash build-selected.sh
bash run4.sh test_selected.py
bash run4.sh final_regression.py
# 回到本机先运行 capture_artifacts.py before；不得同时启动推理
bash run4.sh formal_selected.py --round 1 --pairs 30
bash run4.sh formal_selected.py --round 2 --pairs 30
bash run4.sh formal_selected.py --round 3 --pairs 30
bash run4.sh short_selected.py --pairs 12
bash run4.sh profile_selected.py
```

`final_regression.py`复核34例输出；单次暖态样本只用于正确性回归，不作P95结论。末层profile的计时包装显式登记为预期forward，包装只调用原保存方法；此变更仅在独立profile脚本内，正式Owner的资格保护不变。worker区间重叠，SDK时间包含提交/等待/转换，不是纯设备时间。

全部实验退出后同步日志，再在本地仓库运行：

```bash
python3 fusion-pipeline-experiment/capture_artifacts.py after
python3 fusion-pipeline-experiment/preserve_service.py after
python3 fusion-pipeline-experiment/validate_records.py
python3 fusion-pipeline-experiment/update_report.py
python3 fusion-pipeline-experiment/check_report.py
python3 -m compileall -q fusion-pipeline-experiment
for f in fusion-pipeline-experiment/*.sh; do bash -n "$f"; done
git diff --check
```

哈希脚本拒绝在测试容器存活时运行，并核对最终验收前后源代码、配置、库、图、原模型权重和公开SDK头文件。`preserve_service.py after`核对线上容器身份/启动时间/配置、554个发布文件SHA及五次完整HTTP响应。最终结论以`logs/summary.json`、`VERIFY.txt`和报告第五阶段为准；历史小样本不能替代正式结果。未覆盖服务并发、多Owner共享和业务准确率。

补充：`screen-precision615.json`保留恢复MLP边界后仍有matched615-1失败的结果。`mixed_bridge.cpp`恢复历史已用过的公开FP32 MatMul输出以检查双重舍入，并为每个句柄固定dtype、使用独立银行。`build-mixed.sh`、`test_mixed.py`、`screen_float.py`是该对照入口。`fp32_owner.py`只在真实B1/T615 logits范围内选择FP32银行，退出后恢复资格；不按请求内容或保存答案选路。末层只算一行时，FP32组合使用原CPU三Linear及BF16中间边界，避免旧115行图的补齐与额外舍入。最终`build-selected.sh`需同时构建mixed桥接，最后的选型记录以execution.json为准。

还核对了更早的native A/C布局尝试，避免只扩shape重复：cpu-tail-experiment/native_bridge.cpp及logs/bridge-screen.json使用公开AC_layout=NATIVE、自有A打包/C解包；当时115行顺序筛选baseline约329.66ms、native约438.11ms，末尾baseline约361.60ms。存在漂移且不是正式配对，不能移作615性能结论，但也没有证据支持把同一方案直接扩shape作为新收益。native-layout-attrs.json保留实际SDK返回布局，不能只依据通用header示例假设stride。当前公开MatMul未提供未量化FP16×FP16直接BF16输出的枚举；带INT4权重的BF16输出类型不符合本任务权重约束。这里不把既有失败或受限类型包装为可用路线。

历史筛选需使用对应logs/*-sources快照和当时execution配置，在另一个独立目录复现；不要把最终selected_owner替代早期研究版本后直接运行旧筛选脚本。logs/final-sources保存实际最终运行源码；数据采集后的本地文档生成器只改变呈现及复核元数据，不改计时程序和原始输出。
