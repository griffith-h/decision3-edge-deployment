# 第六阶段：BF16融合与跨层流水（验收完成，1秒未达成）

目标是615 token暖态模型方法P50/P95都<1秒，原current精度门槛不变。对照是第五阶段已通过34例的selected候选，不把第四阶段validate_mlp当作新的合格起点。线上冻结服务保留，未部署、未提交、未推送。

## 运行隔离

板端 `/opt/decision3-edge-bf16-pipeline-20261018`；无端口容器 `decision3-bf16-pipeline-test`。`run.sh`按固定镜像、CPU4–7、8GiB、Torch3线程、扫描上限4执行，所有历史目录只读；第五阶段二进制通过 `/fusion` 只读挂载复用。每次启动前保存当前源码快照。必须等待容器退出才上传或构建；正式计时不与hash、profile、编译、HTTP检查重叠。

先执行本地 `python3 bf16-pipeline-experiment/preserve_service.py before`；收尾执行after以核对线上身份、554个发布文件SHA及HTTP输出。构建脚本拒绝与测试容器重叠。

## 历史核对与新机制

- 原快速MLP图省略BF16中间边界，在新615内容上未过0.01门槛。恢复MLP边界但仍用半精度C也不足，故本阶段沿用FP32 GEMM输出。
- 历史native A/C为115行FP16 C，整体更慢。本阶段不是扩shape重试：新FP32 native C布局从公开SDK实际查询确认，直接融合其C缓冲与down的native A，避免完整gate/up普通布局解包；仍需独立实测。
- `probe_cast.py`：固定Toolkit2 2.5.0中BF16 Cast构建因NumPy类型错误失败；half/round控制可构建。公开`rknn.py` config文档只列float16非量化类型。此证据仅限当前构建路径，不代表所有SDK/编译器都不能融合BF16。
- `fused_boundary.cpp`：保持原FP32->BF16、BF16 SiLU、BF16乘法->FP16输入，减少临时数组；`test_boundary.py`逐位核对及NaN/Inf/65496饱和、乘积溢出等保护。缓存组件慢于原实现，未据此认领整模型收益。
- `stream_scan.c`：原32行block/64列tile，完整FP32状态跨分段传递；每段全部head完成才发布。`test_stream.py`包含615尾块并要求逐位一致。初版导入路径失败单独留存日志。
- `stream_pipeline.py`：一个扫描生产者和一个后续norm/out_proj/residual/MLP消费者。仅固定B1/T615、零初态、无cache、无mask且资格完整的线性decoder层；原输入准备、原全部token和扫描状态保持。所有输出私有，全部worker结束后才提交。
- `cross_layer.py`：在前一线性层行完成后提前下一全注意力层逐行norm/Q/K/V，只有精确返回的hidden对象能在紧接着的该层消费私有结果；完整615行K/V齐备后才进入非因果注意力。最后私有readout层不预取，避免无用计算。

## 独立执行顺序

在板端目录运行；每条命令完成后才运行下一条。日志全部保留。

```bash
bash build.sh
bash run.sh test_stream.py
bash build-boundary.sh
bash run.sh test_boundary.py
bash run.sh screen_pipeline.py --group 128 --threads 3 --pairs 6
bash run.sh screen_pipeline.py --group 256 --threads 3 --pairs 6
bash run.sh screen_pipeline.py --group 512 --threads 3 --pairs 6
bash build-native-probe.sh
bash run.sh probe_native_attrs.py
bash build-native.sh
bash run.sh test_native_mlp.py
bash run.sh screen_cross.py --group 256 --threads 3 --pairs 6
```

screen仅为筛选；每模式两次暖机、六组交错，全部原始输出保存，不挑慢样本，不作正式P95验收。MLP真实回放使用只读 `/mlp/mlp-layers.pt` 原权重及 `/tail/captured.pt` 真实615激活。原误用 `/work/mlp-layers.pt` 的失败另有日志，不能算通过。

若新路径胜出，须另做34例原current精度、故障/回退/生命周期、无插桩暖态正式交错配对及独立profile；组件逐位一致或单个内容过门槛都不能替代这些验收。

## 固定模型结构核对

实际`text_config.layer_types`为`[linear, linear, full, linear, linear, full, linear, linear, full, full, full, full]`，不是严格交替。跨层full预取目标为1→2、4→5、7→8三处。初版测试误把六线性/六完整注意力当作严格交替，错误期望5次命中；实际3次命中且输出门槛通过，保存`screen-cross-transition-count-error.*`后按实际目标集合修正测试，未修改模型层序。没有把仅暖机的失败运行算作性能证据。

## 最终采用路径与验收

采用`accepted_owner.Owner`（`split_owner.py`），由`selection.json`记录：仅B1/T615 logits，公开未量化FP16×FP16→FP32、native C→A原BF16边界融合；down以mask1/2处理320/295行，原次序拼回615行。其他投影沿用第五阶段normal FP32，扫描保持原32/64、4线程，流水和预取未采用。上下文分别持有独立A/B/C缓冲，全部归新bank所有；旧engine/pool只借用，不能用旧mixed ABI销毁native bank。

```bash
bash build-selected.sh
bash run.sh test_native_ranges.py
# 回到本机：python3 bf16-pipeline-experiment/capture_artifacts.py before
bash run.sh regression_native.py
bash run.sh formal_native.py --round 1 --pairs 30
bash run.sh formal_native.py --round 2 --pairs 30
bash run.sh formal_native.py --round 3 --pairs 30
bash run.sh short_formal.py --round 1 --pairs 12
bash run.sh profile_native.py
# 全部退出后在本机：
python3 bf16-pipeline-experiment/capture_artifacts.py after
python3 bf16-pipeline-experiment/preserve_service.py after
python3 bf16-pipeline-experiment/validate_records.py
python3 bf16-pipeline-experiment/update_report.py
python3 bf16-pipeline-experiment/check_report.py
```

正式90次候选P50 1579.62ms、P95 1642.71ms，原34例门槛通过，相对上一合格候选最大受检数值差0.00000047。原current参照来自只读第五阶段34份独立current，验收哈希包含参照文件；本轮另新算615 current并断言相等。输出没有作为推理缓存使用。范围/故障/关闭详见VERIFY和原始JSON。

正式前完整JSON相等误断言保存为`formal-baseline-bitwise-assertion-error.*`，没有产生正式样本。旧第五阶段回归与正式输出已有约1e-7浮动；身份附加检查改为全部数值≤1e-6、choice一致，实际完整baseline输出和delta先保存后断言。原current的0.01概率/score/noul门槛从未放宽，confidence仍独立报告。所有正式样本保留；组件漏暖机初版归档`split-mlp-initial-cold-control.*`，不能混入暖态指标。

SHA扫描仅在测试停止后运行；失败身份断言前后323项资产相同，保存为`artifacts-baseline-assertion-aborted-*`。重新冻结后最终329项验收资产前后相同。各运行的`logs/sources/`保存当时源码；本地报告/校验后处理可能晚于数据采集，不能替代实际运行快照。未覆盖多Owner并发、冷启动SLA、任意业务输入或业务准确率。
