# 本轮验证记录

2026-10-19。以下为实际执行，不代表完整1,000条或发布验收。

| 检查 | 命令 / 证据 | 结果 |
| --- | --- | --- |
| r011完整开发验证 | `check_validation.py`，`iterations/r011/20261019T102243-validation-batch-mlp-projection-fp32-scan-npu-attention-readout-plain/` | 8请求、34回归、20保护配对通过，关闭四类句柄0，原服务恢复健康 |
| r012长输入重复 | `execute.py --iteration r012 --case long7232 --variant batch-mlp-projection-fp32-scan-npu-attention-readout --repeat 3 --budget 120` | 59.53/56.07/58.27s，中位数58.27s；`check_case.py`通过，长→615/115切换通过，句柄稳定并关闭0 |
| 新慢请求诊断 | `execute.py --iteration r012 --case coverage --variant batch-mlp-projection-fp32-scan-npu-attention-readout --budget 30` | 首条18问题预算内未完成，专用容器清理，原服务恢复健康；不是600秒超时 |
| 新慢请求profile | 同版本，`--case coverage-0000 --profile --budget 120` | 完整88.23s，注意力独占39.55s；用于定位，不当性能基线 |
| 新慢请求无包装对照 | 同版本，`--case coverage-0000 --repeat 2 --budget 120` | 87.93/88.02s，输出重复一致，原服务恢复健康 |
| r013公开SDK库构建 | 板端 `/opt/decision3-official-opt-20261019-r013/build_npu_attention.sh` | 编译成功，见该轮`build.log`；新域实际SDK测试尚待执行 |

原始路径均在忽略的 `output/decision-index-audit/`，不提交题目/标签/完整输出。

本机定向命令：

```sh
output/decision-index-audit/venv/bin/python -m unittest discover -s official-long-input-experiment -p test_coverage_scan.py -v
# 3项通过：慢请求停止、异常恢复、缺字段/非有限结果。
output/decision-index-audit/venv/bin/python -m unittest discover -s official-long-input-experiment -p test_long_input_owner.py -v
# 3项通过：安装/关闭顺序、部分安装失败、清理错误与原始错误保留。
output/decision-index-audit/venv/bin/python -m unittest discover -s official-long-input-experiment -p test_summarize.py -v
# 2项通过：noul跨0.5阈值即拒绝，小数值差不能掩盖判定变化。
output/decision-index-audit/venv/bin/python -m unittest discover -s official-long-input-experiment -p test_regression_gate.py -v
# 3项通过：原current的0.01门槛不变，中长度变化报告双参照，115/615仍要求native身份一致。
output/decision-index-audit/venv/bin/python -m unittest discover -s official-evaluation -p test_verify_http.py -v
# 2项通过：HTTP保留images/videos且不带期望输出，noul判定一致。
python3 -m py_compile official-long-input-experiment/*.py
python3 -m py_compile official-evaluation/verify_final_http.py
git diff --check
```

上述语法与差异检查通过。实际SDK/数值/性能在RK顺序执行；未以本机mock替代新候选板端检查。统一工厂、完整样本评测和最终8901/8900发布仍待真实验证。

## r014～r018：正确性诊断

- r014/r015均实际执行固定ARM镜像32项投影/MLP/readout/scan/NPU注意力检查，通过；这不等于真实新题组通过。r014开发120秒未完成；r015完整profile90.05秒，`home-appliance-check.json`记录数值失败，不采用。
- `execute.py --iteration r016 --case ablation-pair --variant full-batch --budget 120`：七种模式完整结束；全部四类组件回CPU与实际B2原CPU输出一致，单独回CPU不足以修复。
- `execute.py --iteration r017 --case ablation-single --variant full-batch --budget 120`：四类组件分别独立启用、再独立CPU完整返回；每项实际命中/旁路经`check_ablation.py`验证，关闭句柄全0、旧服务健康。16项公开SDK合成FP16范围探针完整精确返回，句柄关闭。
- `python3 -m unittest discover -s official-long-input-experiment -p test_ablation_suite.py`：2项通过（禁用状态、异常恢复）。
- `python3 -m unittest discover -s official-long-input-experiment -p test_cpu_batch_reference.py`：3项通过（原8/8/2几何、意外NPU、准备token变化）。
- `execute.py --iteration r018 --case coverage-0000 --variant cpu-full --budget 600`：推理完整结束，原8/8/2批次329.32/319.93/79.37秒，完整18问题与20,051token；四类加速调用增量均0，关闭句柄全0。合计728.62秒仅为正确性参照，非性能基线，服务恢复仍由控制器核验。原B8与逐行B1参照最大差0.043934、choice相同；对原B8重新核对r015仍差0.076166并改变choice，原失败结论成立。
- `component_observer.py`及3项仪器生命周期检查已准备，板端r019尚待执行；本机仅完成语法检查，未把准备状态写成真实推理通过。

所有负面输出、旧逐行参照与旧检查结果保留。新原批次参照的结论使用独立文件和参照SHA，不覆盖旧证据。

## r019～r022：同输入组件与线性精度诊断

- r019 `--case ablation-components --variant full-batch --budget 120`：32项ARM边界检查与3项观测仪器检查通过，五种模式最终输出与r017未观测版本逐项相同；所有观察包装反序恢复、四类句柄0、原服务健康。`check_ablation.py`与`observation-review.json`已复核。真实首层注意力对原BF16 SDPA约11.55%元素不同，对FP32替代约0.0124%；扫描约0.0294%、MLP约0.590%。这些是组件差异，不是任务准确率。
- r020 `--case diagnostic-home-pair --variant full-batch-precise-medium --budget 120`：35项ARM边界检查通过，中长度原CPU注意力与扫描实际命中（native计数增量0），完整两问题15.59秒；对实际B2原CPU最大差0.092489且choice变化，**拒绝采用**。关闭句柄0、旧服务健康；只修两类组件不足以通过完整模型数值门槛。
- `python3 -m unittest discover -s official-long-input-experiment -p test_medium_precision_candidate.py`：3项通过，覆盖原批次输入身份、115/615/长输入旁路、异常恢复原enabled集合对象。
- r021自有公开SDK FP32输出读取库编译成功（引用的旧自有源码有两条缩进警告）；32项原组件检查通过。新增微型布局用例K16被实际SDK拒绝，要求K按32对齐，真实模型探针未执行。错误日志、源码快照及`probe-failure-review.json`保留，四类句柄0、旧服务健康。
- r022仅把微型布局用例改为K64/N32，复用相同编译库和实际模型K1024，探针**运行中**。该探针刻意在第一个投影后停止，不等同完整模型推理或准确率成绩。

实际板端命令均由`caffeinate -i output/decision-index-audit/venv/bin/python official-long-input-experiment/execute.py`调用。尚无新版本发布。

## r023～r028：临界位置修复，尚无端到端验收

- r023真实投影各256个有意义的舍入差异位置：原CPU4×4/8×8/16×16/32×32 GEMM与原FP32逐项相同；逐点计算不相同。
- r024～r026累计同两个完整512行投影约419万输出：修复后受检BF16差0。r026原CPU整块诊断0.229/0.081秒，修复0.534/0.190秒；不能把组件数值一致冒充速度收益。r024/r025分别执行5项CPU修复检查，r026执行6项（新增尾部tile补齐），全部通过。关闭句柄0，原服务恢复健康。
- r027 `cpu_kernel_preflight.py`：首次脚本别名错误已保留并修正；第二次直接公共CBLAS与Torch合成GEMM差异，拒绝进入模型。未暂停线上服务，无新模型推理。源码快照、两次stderr及`decision.json`保留。
- r028 `build_sparse_cpu.sh`：自有native行搬运库编译成功。CPU容器内 `python /lab/cpu_kernel_preflight.py`：3项搬运检查通过（跨行stride/非连续索引/尾tile、边界与dtype拒绝、空输入），新旧行搬运的TorchBMM结果相同。该CPU预检没有加载模型或NPU，且未包`inference_mode`，其中计时不是模型推理性能成绩。
- r028 `execute.py --case diagnostic-home-pair --variant full-batch-sparse --budget 120`：**运行中**，完整请求、49项相关检查与关闭结果尚待返回，不提前写通过。

## r028～r031：完整诊断请求与准备融合

- r028两问题完整90.07秒、最终全部数值差0，但12次投影运行回退暴露“私有inference权重无版本计数器”，因此不采用。实际49项检查通过；关闭0、原服务健康。`integration-review.json`记录负面结果。
- r029加明确的原参数守卫授权：仅私有打包权重用identity缓存，原模型权重仍校验版本；新增相应回归，50项检查通过。完整profile96.78秒、最终数值差0、运行错误与运行回退0。修复内部范数13.90、筛选11.75、搬运8.24、小矩阵38.44秒，为互斥计时，不能当正式性能。
- r030单独测试公共CBLAS临界位置修复，原整模型0.01/choice/noul门槛保持。已知合成16,384输出中1个与Torch不同的证据保留，不作逐比特替代声明。实际53项检查通过；固定两问题完整57.636秒，最终所有受检数值含confidence均差0、回退与错误0、关闭0、原服务健康。仍不是完整题组/完整测试集或正式配对性能验收。
- r031 `build_sparse_cpu.sh`：加入自有范数/FP16误差准备与同一FP32筛选规则，`-ffp-contract=off`，编译成功。CPU容器 `python -m unittest -v test_native_preparation_candidate`：3项通过，含小值/跨行stride、同Torch筛选与顺序、非有限/密度/系数与范围拒绝；没有加载模型或NPU。
- r031 `execute.py --case diagnostic-home-pair --variant full-batch-sparse-native --budget 120`：**已启动**，56项相关集成检查及完整结果待返回。全部1,000条和部署仍暂停。

## r031～r033：真实请求与新候选

- r031 `execute.py --iteration r031 --case diagnostic-home-pair --variant full-batch-sparse-native --budget 120`：56项实际检查通过；35.901秒，全部最终数值差0、回退/错误0、关闭0、旧服务健康。完整18问题随后`--case coverage-0000 --profile`超过120秒开发预算，未完成、无最终输出，不能计准确率；负面记录与清理保留。
- r032实际SDK `python -m unittest -v test_fused_sparse_mlp_candidate`：跨库native布局、BF16乘积、320/295拆分、补齐及范围/越界拒绝通过。整模型相关57项检查通过；两问题36.300秒，数值差0、回退/错误0、native句柄48且关闭0、旧服务健康。相对r031没有速度收益，仅在此请求观察到更少句柄和较低RSS。
- r033新库`build_npu_attention.sh`编译通过。新增中长度CPU-Flash语义边界候选，真实SDK定向测试正在执行，未部署。

- r033 `execute.py --case diagnostic-home-pair --variant full-batch-sparse-flash --budget 120`：61项相关检查通过；完整33.478秒，但对同B2原CPU最大差0.11721915且choice改变，**拒绝采用**。五类句柄关闭0、原服务恢复健康。原始源码与失败输出保留。
- r034控制器首次因本地迭代目录尚未创建而退出，发生在停服务/建worker之前；补齐独立目录与原输入清单后重新启动。新增CPU线性分块候选正在检查，尚无结果。

- r034新增两项CPU边界检查中，FP32线性转BF16的逐项相同检查失败，完整模型候选未执行，五类句柄关闭0、原服务恢复健康；没有改宽单测继续跑。r035另行CPU-only诊断原BF16行块与FP32路径的差异/成本。

- r035 `cpu_dense_preflight.py`在原ARM镜像CPU-only执行：BF16切块输出0差但无提速；FP32快约8倍但有344/2,710处BF16舍入差异，不能宣称无损。
- r036 `build_sparse_cpu.sh`编译通过；`original_repair_preflight.py`的3项定向检查中1项失败（K1024/N128的1个输出不一致），因此没有执行候选模型/完整请求；失败源码和日志保留。正在核对实际CPU派发与固定公开源码归约，未扩大测试。

## r037～r040：纠正合成参照，回到实际CPU路径

- r037公开BF16 BLAS的3项合成检查通过，后续预检因合成层在inference_mode内创建、没有版本号而退出；未放宽真实权重守卫。r038只修正fixture创建方式，约367万不同位置输出在两次测量修复后与原生Torch BF16逐项一致。但这不是本仓库当前CPU的矩阵参照，不能据此声称模型收益。
- 源码核对：`cpu_optimized_server.optimize`给原模型Linear缓存FP32权重并安装`linear_forward`；桥接`_bp_original`捕获该实际forward。r034的原生Torch BF16单测选错参照。r039已在候选推理前SIGINT安全停止，控制器清理、原容器健康，证据`reference-correction.json`。原整模型r017/r018参照保持，不改写既有数值失败/成功。
- r040修正参照为实际`cpu_optimized_server.linear_forward`，复用同一缓存FP32权重并按512行内部切块；正在跑原严格逐项检查和同B2完整请求，再以冻结r017输出核验。没有额外BF16 BLAS修复。

- r040 `execute.py --case diagnostic-home-pair --variant full-batch-dense-medium --budget 120`：37项相关实际检查通过（含实际`cpu_optimized_server.linear_forward`尾块/形状/私有输出/恢复）；54.460秒、所有最终数值差0、投影/MLP回退错误0，关闭0、原服务健康。慢于r031，不采用为速度优化。
- 清理错误参照相关原型：移入忽略目录`rejected-bf16-reference-experiments/`并保存源码SHA；原板端运行快照及各轮负面证据保留。当前Python接口不提供这些被撤回的候选。
- `python -m unittest ... test_execute.py`1项、`test_host_run.py`2项通过；Python语法与`git diff --check`通过。

- r041 `row_blas_preflight.py`仅CPU：1×4在两种形状分别6/4个BF16输出不同，拒绝；4×16输出相同但0.047/0.299秒慢于原0.040/0.115秒，拒绝，未启动模型。
- r042新稀疏准备库编译完成：新增同FP32运算顺序的NEON筛选，保留原scalar比较路径；4项边界/尾部/阈值检查和同行4×4预检正在CPU-only执行。

- r042 CPU-only4项检查通过，NEON/scalar选点12,810项逐项相同；同行4×4受检输出相同但只获小幅组件改善，尚未进入模型。
- r042 `execute.py --case diagnostic-home-pair --variant full-batch-sparse-vector --profile --budget 120`：60项实际相关检查通过；完整profile35.301秒、全部最终数值差0、4,229,571选点/105,984改变数与r031相同，回退与错误0、五类句柄关闭0、原服务健康。修复总13.056秒，互斥范数1.833/筛选1.897/小矩阵8.994秒；不把该profile和单次plain差值当正式配对收益。
- r043稀疏对角项实验库编译完成，只使用自有NEON及公开数学累加顺序；正在CPU-only对固定公共4×4 BLAS核对352/1024块，不加载模型。

- r043 `diagonal_preflight.py`CPU-only：K64/1024/2048/6144/10240、共86,567选点，Q352对此前公共4×4修复输出逐项一致，未选位置不变；Q1024出现6/5/5处不一致，不采用。该结果只证明固定测试矩阵，非任意输入的数值保证。
- r044新快照修正自有C缩进警告，库编译完成；`execute.py --case diagnostic-home-pair --variant full-batch-sparse-diagonal --budget 120`已启动，原矩阵/输入/选点与门槛保持。完整模型结果待返回，未提前采用。

- r044两问题：63项实际相关检查通过；29.2176秒、原CPU实际同B2输出全部数值差0、choice/noul/usage/model一致，选点/改变数与r031相同、回退/错误0、句柄关闭0，原服务健康。内部修复7.255秒；单次约18.6%改善，不是正式配对性能结论。
- r044 `execute.py --case coverage-0000 --variant full-batch-sparse-diagonal --budget 600`已启动：一次完整18问题的独立完整性/正确性核验。原数据、8/8/2几何、20,051token和原0.01/choice/noul门槛不改；目的及CPU参照SHA已在运行前记录，120秒速度目标仍保留，未提前写通过。

## r044完整题组OOM与r045内存修复

- r044完整18问题未返回，exit137、原容器健康；Docker事件的OOM与kernel的同一容器`ac3c7144...`cgroup互相核实。8GiB已耗尽，记录为OOM，不是600秒超时或完整准确率。原始`oom-kernel.log`/`oom-docker-events.jsonl`和该运行`failure-review.json`已保存。
- r045 `packed_cache_policy.py`只释放私有打包模块的派生FP32后备缓存，原CPU数学按需重建；原模型参数与原CPU缓存不变。CPU-only `python -m unittest -v test_packed_cache_policy`3项通过：派生缓存引用释放、后备结果相同、未来打包只处理一次、未知函数旁路、工厂恢复和无模块强引用环。
- 受控worker现在先保留退出状态给控制器读取OOMKilled/ExitCode，再按原所有权检查移除；CPU-only预检保留--rm。`test_host_run.py`3项、`test_execute.py`2项通过（含OOM状态落盘与原服务恢复）；语法/差异检查通过。
- r045 `execute.py --case diagnostic-home-pair --variant full-batch-sparse-compact --budget 120`已启动；组合原Q352精度修复、已验证双down融合及派生缓存退役，先检查再跑整模型。内存改善的数值和是否消除OOM尚待板端结果，不提前写通过。

- r045两问题实际67项检查通过；28.9097秒、最终数值含confidence差0、choice/noul/usage/model一致，运行错误/回退0、生命周期0、原服务健康。24个私有模块释放派生缓存312MiB，私有CPU调用/守卫旁路0；native上下文48、额外单down0。cgroup该请求峰值6,165,147,648字节、OOM/oom_kill0。退出状态已实际落盘为exited/ExitCode0/OOMKilledFalse，清理完成。
- r045同完整18问题600秒核验已启动，提前记录用途及固定CPU参照；仍是8GiB/原8-8-2批次/全部20,051token及原数值、离散门槛，不把120秒速度目标放宽。完整结果、题组指标及OOM是否消除待返回。

- r045完整18问题真实返回：243.6759秒、20,051token、原8/8/2几何，运行错误/回退0；内存峰值8,170,962,944字节、OOM/oom_kill/max0，生命周期0、退出OOMKilledFalse、原服务健康。原生字段正确率与CPU同为11/18，但非confidence最大差0.04244779>0.01，原门槛失败，120秒速度目标也未达成；已保存完整`home-appliance-check.json`失败结果。
- 依据完整结果暂停纯速度优化，r046准备原B8首个投影对实际CPU forward的逐项诊断。故意只执行部分模型，不将诊断完成标成完整请求或任务准确率。

- r046 `--case first-full-projection --variant full-batch-sparse-compact --budget 120`：原8×1181的qkv/z两投影共77,398,016输出，对同一输入实际当前CPU forward全部一致；NPU组诊断1.740秒、CPU两投影3.982/1.307秒仅用于数值。故意只执行首投影并结束，无完整模型准确率。
- r047准备CPU融合首差异探针，参考从包装函数的`_old`捕获实际CPU函数；输出原样返回，遇首差异或第一层结束停止，全部方法finally恢复。

- r047原8×1181首层四处CPU融合观察（inputNorm、in_proj_b/a、gatedNorm），对捕获的实际前一CPU函数输出逐项一致；诊断只完成第一层，无完整模型指标，关闭0、原服务健康。
- r048 CPU-only `test_medium_cpu_original`3项通过：同一序列/计数对象，保护及长域旁路，异常精确方法恢复，未知方法守卫。
- r048 `--case first-cpu-layers --variant full-batch-sparse-compact --budget 120`已启动：同输入解码层CPU融合消融，NPU线性/MLP相同，首差异或前三层相同即停止；每层落盘，不当完整模型准确率。

- r048CPU融合解码层消融前三层均逐项一致（原B8、67个捕获的CPU方法、相同NPU线性/MLP）；不代表真实全CPU前缀一致。部分模型记录`cpu-layer-observations.json`，未计算准确率。
- r049准备候选与实际owner.current CPU前三层前缀对照；撤销开发适配，强制原CPU线性，CPU段要求所有NPU/图计数增量0。两个部分模型调用分别受120秒预算约束，明确不是完整请求成绩。

## r049 / r050：真实CPU decoder前缀诊断

r049 `execute.py --iteration r049 --case cpu-prefix-reference --variant full-batch-sparse-compact --budget120` 完成，原B8/T1181初始hidden及前三层逐项一致，CPU NPU/graph增量0，关闭0、原服务恢复健康。候选28.24秒/CPU49.26秒仅部分模型诊断，不算完整速度。

r050独立快照运行 `execute.py --iteration r050 --case cpu-decoder-reference --variant full-batch-sparse-compact --budget120`：候选第一批12层后停；CPU每三层一个诊断预算，传递CPU自己上层hidden、保留原mask/位置/完整B8。结果待实际运行。静态检查 `python3 -m py_compile official-long-input-experiment/cpu_decoder_probe.py official-long-input-experiment/probe.py` 通过。

r050实际完成：前4层全部BF16一致，第5层180个不同/maxΔ0.0009765625。CPU replay每层后NPU/graph计数未增加，退出关闭0、原服务健康；r018/r045 identity所有文件SHA无差异。r051 `execute.py --iteration r051 --case layer4-components --variant full-batch-sparse-compact --budget120` 定位该层投影与MLP内部首次偏差；静态py_compile通过，结果待运行。

r051实际完成8项原B8第5层组件比较：z投影唯一一处BF16不同，qkv/out/MLP gate/up/乘积/down/result均逐项一致；故意停止、关闭0、原服务健康。r052 `execute.py --iteration r052 --case projection-single-point --variant baseline --budget120` 为保存单点的CPU-only诊断，尚待结果。`cpu_point_probe.py`及`probe.py` py_compile通过。

r052单点CPU-only重放未复现保存的CPU输出（1,997处BF16不同），主动失败，原服务恢复健康；不能据此采用任何单点修复。r053先加载相同实际Owner但不预热/不装候选，检查原CPU函数源码、原权重/缓存与checkpoint一致性、线程/BLAS上下文，再对照保存输出；前后NPU调用必须无增量。其complete事件的`npu_owner_loaded:false`是诊断标记漏改，实际加载Owner由该run源码和lifecycle记录证明；本地已修正，不能按此标记声称完全未加载。

r054实际诊断：原点未进入经验带，raw到中点距离1.66893005e-6 > band1.45360718e-6；同上下文单点Q352/CBLAS均得到原B8CPU值。原CPU row1该点不同于完整B8，再次保留CPU几何依赖证据，不以row1替代参照。关闭0、原服务健康。r055 `execute.py --iteration r055 --case cpu-prefix5-reference --variant full-batch-sparse-band2 --budget120` 新系数2e-7、原密度/点数上限及门槛不动，2项针对检查+原B8前5层CPU对照；py_compile及diff --check已通过，实际结果待运行。

r055前缀数值诊断实际完成：69项相关ARM检查通过，初始hidden和前5层BF16逐项一致，CPU NPU/graph增量0，关闭全0。后续两问题启动过早与恢复等待重叠，准备阶段主动SIGTERM中断，无完整结果；双方退出后同原服务健康，双方附controller-coordination-review.json，不计性能/精度。新增控制器全程flock覆盖恢复期；`python3 -m unittest discover -s official-long-input-experiment -p 'test_execute.py' -v` 3项通过（含恢复期拒绝第二控制器），原host_run 3项仍已通过。现待干净串行重跑同板端r055两问题源码。

r055两问题干净重跑30.8578秒，CPU r017同B2参照全部数值含confidence差0，错误/回退/密度拒绝0，关闭0、原服务健康、OOM0；pair-cpu-check.json记录30秒速度目标未达。原源码同iteration开始 `--case coverage-0000 --variant full-batch-sparse-band2 --budget600` 一次完整性核验，complete-case-purpose.json已在运行前记录；不把600秒换成新速度目标。

r055完整18问题实际255.3375秒，原8/8/2、20,051token完整返回，字段正确率11/18同CPU、choice/noul一致；非confidence最大差0.04268233/含confidence0.08333921，仍失败。60,464,352选点/934,316改变，密度/错误/回退0，峰值8,189,042,688字节、OOM0，关闭0、原服务恢复健康。home-appliance-check.json保存failed；不部署。之后才串行启动r056 `--case cpu-decoder9-reference --variant full-batch-sparse-band2 --budget120`，部分模型后续首差诊断，静态py_compile通过，结果待实际运行。

r056前9层实际全部BF16一致，CPU每层NPU/graph无新增，关闭0、原服务健康，候选88.80秒仅诊断。接续同r056 `--case cpu-decoder-reference --variant full-batch-sparse-band2 --budget120` 检查完整第一批12层，仍相同原模型/B8/输入，候选120秒，CPU每三层120秒分段，仅诊断，不篡改正式协议。

后续诊断辅助代码（本地已准备，未写入正在运行的r056）：cpu_decoder_probe.py在首差时保存该层input/候选/CPU输出及原mask/位置；若12层一致，则保存CPU末层hidden并按原norm、FP32 readout、温度、softmax和respond重建原第一批8题的CPU诊断响应。它不是完整18题结果，也不用于生产输出缓存。py_compile通过；实际运行未验证前不得当作已取得的新CPU参照。

r056完整第一批decoder已完成：前11层逐项一致，第12层1个BF16值不同/maxΔ0.25。CPU每层NPU/graph无新增，关闭0、原服务健康；候选114.46秒仅部分模型诊断。输入身份复核r018/r045/r055/r056 coverage文件及单case规范化SHA相同。r057 `--case cpu-decoder-reference --variant full-batch-sparse-band2 --budget120` 引擎不改，诊断保存首差激活/位置；CPU改与原encoder相同__call__，末层后以同一原readout重建CPU/候选首B8诊断响应，核验原r018第一批。py_compile通过，真实结果待运行；不是新的完整准确率或部署验收。

r057完整decoder/head诊断：前11层一致，末层唯一差点row4764/column0不在最后位置；CPU重放与候选hidden的同原head答案差0，且同r055实际首B8差0，均与r018旧CPU首批不同。此为CPU参照/流程一致性疑点，不能改数值门槛或直接替换旧参照。关闭0、原服务健康。r058完整复制r018原顶层文件与inputs，无新候选/新测试，`execute.py --iteration r058 --case coverage-0000 --variant cpu-full --budget600` 独立复核原model.system_one完整18题原8/8/2各批，保留旧结果；只作正确性，不作性能基线，不是17阶段恢复。

r058全部18题原CPU正常调用与r018所有数值含confidence差0，三批NPU/graph增量0，顶层lab-sources源码/二进制SHA相同、关闭0、原服务健康；cpu-reference-reproducibility.json记录通过，旧参照继续保留。r059切换审计：`--case cpu-switch-audit --variant full-batch-sparse-band2 --budget600`，候选事件明确收紧120秒、CPU正常首B8正确性600秒；不包encoder/decoder、不重放hidden，记录实际token指纹/内核链与公开CPU完整接口结果。辅助控制器request_budget只允许收紧、不允许放宽，非法值拒绝；最近 `python3 -m unittest discover -s official-long-input-experiment -p 'test_execute.py' -v` 4项通过，py_compile通过。r059结果待实际运行，不能提前当作完整精度通过。

r059实际正常CPU model.system_one首B8与候选全部数值差0、实际token指纹/counts相同；与旧CPU仍不同。CPU NPU/graph增量0、关闭0、原服务健康，switch-consistency-review.json保留。r060 CPU-full后端状态审计四阶段MKLDNN True，候选r059此前却False；其余Torch3/highest/autocastCPU False。准备同源band2纯读取逐步骤审计，未改任何后端值或原门槛。


## r061 / r062: isolated checks and original backend reference

r061 ran all 69 ARM boundary checks in one separate child process. The child ended with MKLDNN disabled after its fault-injection test, while all 20 parent model snapshots stayed enabled. The original enabled-backend B2/T1193 CPU diagnostic was saved separately (94.4899s, correctness-only, NPU/graph delta zero).

r062 same-factory two-question candidate completed in 33.2845s. Against this enabled-backend reference, maximum numeric delta including confidence is 0.007825390976162583; choice/noul/usage/model unchanged. Original 0.01 numeric gate passes; 30s speed goal fails. 69 child checks pass; parent backend remains enabled through actual inference, errors/fallback/density rejection zero, lifecycle zero, no OOM, retained original service healthy. This incomplete benchmark group cannot establish accuracy or release qualification. Evidence: iterations/r062/canonical-pair-check.json.

Commands: python3 -m unittest discover -s official-long-input-experiment -p test_execute.py -v (4 pass); same with test_isolated_checks.py (1 outer test passes, expected failing child proves failure propagation); ARM child suite (69 pass). Original full CPU r018/r058 and all negative candidate records remain unchanged.


r062 original B8 first projection: 8,340 qkv and 3,023 z BF16 mismatches out of 77,398,016 outputs against actual enabled-backend original CPU, maximum deltas 0.03125 / 0.015625. Partial diagnostic only, not a full-model gate failure or accuracy score. Parent backend True, lifecycle zero, original service healthy. This differs from old False-context r046 and is retained separately. r063 isolates original CPU 3D/flat/selected reduction geometries on these intermediate positions; no production repair is changed. Static py_compile passes.


r064 real 512 mismatching qkv positions: native Q352 differs at all 512; native Q1024 matches all 512, as do enabled-backend Torch BMM4/16. These are only selected intermediate positions. r062 complete enabled-backend case then ran once under the recorded 600-second correctness purpose: 275.2473s, all 18 questions / 20,051 tokens / original8/8/2, field accuracy11/18 and case-exact0 same as CPU, choices/noul unchanged; maximum non-confidence delta0.0426819921 (FAIL), including confidence0.0833386183. No error/fallback/density rejection, memory.peak8,110,141,440 bytes with max/oom/oom_kill0, lifecycle zero and original service healthy. Parent backend True through actual inference. home-appliance-check.json preserves failed, not deployed. The 120-second speed target also fails. r065 CPU-only enabled-reduction-preflight checks all relevant K dimensions before any new arithmetic candidate is accepted; no model or labels loaded.


## r065-r068: continuous-K repair for the original enabled CPU backend

r068 two-question normal request completed33.5495s; max numeric delta including confidence0.00174460295 against original True-backend r061, choice/noul/usage/model unchanged. Original0.01 gate passes, 30-second speed fails. Read/repair errors, fallbacks, density rejection, OOM and lifecycle zero; original service healthy. Same source now runs one complete18-question check under the recorded600-second correctness purpose, preserving original8/8/2 and120-second speed target. Results pending; do not declare full qualification from this pair.

r065 rejects Q1024 as a general repair: larger K dimensions still differ. r066 was a failed build attempt (runtime image has no gcc); source/library hash verification rejected the stale library before model loading. No candidate model result exists for r066. r067 rebuilt using the same board gcc14.2 and unchanged flags as previous sparse builds. A new permitted block16384 exceeds the existing maximum K10240, so the owned dot loop never restarts its FP32 accumulator. All 96,529 selected positions across K64/1024/2048/3584/6144/10240 match the original enabled FP32 CPU; Q352/Q1024 fail some dimensions. Unselected entries stay untouched. CPU-only component evidence does not qualify a model.

r068 adds enabled_sequential_repair_candidate.py after the previous band2 stack. It checks original MKLDNN=True, Torch3 and highest FP32 precision at install and every repair call, rejecting changes before output mutation. Two focused ARM checks exercise original3D FP32 parity, untouched positions, backend-change rejection and installer restoration; all71 ARM checks pass in the isolated child. Actual originalB8/T1181 first qkv+z group (77,398,016 outputs) now matches original CPU BF16 at every value. Full-request qualification remains pending. Local py_compile, test_execute.py4, test_host_run.py3, test_isolated_checks.py1 and diff --check pass.


r068 complete18-question result:275.2284s, all20,051tokens and original8/8/2, field accuracy11/18 and case-exact0 same as CPU, choice/noul/usage/model unchanged. Max non-confidence delta0.0131416686 and including-confidence0.0280218448: original0.01 gate FAILS, speed120 also fails. Only one of18 final answers exceeds the non-confidence numeric gate; it is in the original firstB8. No read errors/fallback/density rejection, memory.peak8,108,470,272 bytes with max/oom/oom_kill0, lifecycle zero, original service healthy. Do not deploy. r069 same arithmetic cpu-decoder3-reference now locates earliest residual difference in original firstB8; three-layer partial diagnostic, not a complete request or accuracy result. Static py_compile passes.


r069 actual partial diagnostic completed: originalB8/T1181 first3 decoder layers each have zero BF16 mismatches versus actual True-backend CPU; CPU NPU/graph increments[0,0,0,0]. Candidate partial34.5866s is not full-request performance. Runtime MKLDNN True, lifecycle zero, original retained service healthy, controller DONE/exit0. This round closes under the planned development budget; no process or automation is left to continue evaluation. Next action is first9-layer localization per CURRENT.md; not yet run. Full18 r068 remains numerically rejected.


## Continuation 2026-10-20: first residual difference after layer3

r069 cpu-decoder9-reference completed its bounded candidate prefix97.8134s; actual original True CPU agrees exactly at layers0/1/2, first differs at layer3 (824 BF16 values, max0.0009765625). CPU loop asserts NPU/graph counters unchanged after each layer. First differing activation is preserved in the ignored run folder. All71 isolated checks pass, backend True and lifecycle zero; original service healthy, controller exit0. r070 same arithmetic layer3-components starts from the complete original request, compares actual original CPU projections/MLP at identical inputs, then intentionally stops. Prefix and same-input component checks are separate120-second diagnostic phases, not full-request timing. Only diagnostic layer-index selection was generalized (0..11), py_compile and diff --check pass.


r070 actual layer3 components: qkv2 BF16 differences (max0.0625), z1 (max0.03125); out-projection and all five MLP stages exactly match actual original CPU on the same inputs. Scope is partial, not a model accuracy result. BackendTrue, lifecycle zero, original service healthy, controller exit0. r071 generic captured-point observer now rechecks all3 expected values using actual original full-B8 CPU before observing live repair; it compares actual selection and forced native/public-BLAS alternatives without changing outputs. It never uses row1 as a canonical reference. Original model and candidate arithmetic unchanged. Static py_compile/diff --check pass.


r071 actual3-point audit reproduces full original-B8 CPU expected values first, then observes live original layer3. All3 are omitted by band2; current bands4.65512e-6/3.83683e-6/7.90624e-6 are below measured midpoint distances4.76837e-6/4.52995e-6/8.58307e-6. Forced existing continuous native dot matches original CPU at all3; public CBLAS alternative does not. Required empirical factors approx2.04866e-7/2.36130e-7/2.17121e-7. BackendTrue, lifecycle zero, original healthy, exit0. r072 adds only rounding_band25_candidate (2e-7 to2.5e-7), preserving density10%, entries65536, model, complete inputs, original batches, fallback and numerical0.01 gates. Two focused ARM tests cover measured selection boundaries and scope/budget restoration; results pending. Local py_compile and controller4 checks/diff --check pass. It is empirical, not a proven universal error bound.


r072 actual layer3-components with band2.5: all3 projection modules and MLP gate/up/product/down/result match original full-input CPU at every value, eight records maximum delta zero. Former qkv2/z1 differences disappear. Full-model numerical qualification remains pending; no speed claim from a partial instrumented diagnostic.


r072 plain two-question request34.4098s; max delta including confidence9.74830259e-6 against r061 original-enabled CPU, discrete decisions/usage/model equal. Original0.01 passes,30-second speed fails. Repair8,063,030 positions, maximum density0.0719346<0.1; errors/fallback/density rejection0, OOM0, backendTrue,73 child checks pass, lifecycle0 and original service healthy. Same source now performs one recorded600-second complete18-question correctness check; original8/8/2/20,051tokens and120-second speed target remain. Results pending. Profiling waits for this gate. Local references.json for potential profiler comparison points only to this same candidate plain run, not old False-context responses.


r072 complete original18-question request285.9999s: maximum non-confidence delta0.00899870409, original0.01 PASSED; choices/noul/usage/model equal. Native field11/18 and case-exact0 equal CPU, home-appliance-check.json passed. Including-confidence max0.01989875267 reported separately, not a new gate. Runtime True/Torch3/highest, 71,241,542 repair selections, max density0.0729674, read/errors/fallback/density rejection0, memory.peak8,113,004,544 bytes with max/oom/oom_kill0, lifecycle zero and original service healthy. Original120-second speed goal FAILS. This is one development group, not complete-suite accuracy or release qualification. r073 same arithmetic profile adds registered-SDPA and normal/native-read versus repair spans; output/counters must match same-candidate r072 plain before using its timings. No performance algorithm changed yet.


r073 profile validated: response delta0 to same-candidate r072 plain, projection/nativeMLP/repair calls/selections/changes and failures unchanged. Source hashes differ only probe.py, runtimeTrue/lifecycle0/original healthy. Exclusive spans: repair10.4247s (norm1.8406, mask2.0666, dot5.9639 nested), scan-layer8.1809, registeredSDPA5.0926, native read/alloc excluding repair2.7039. This is one profiled2-question request, not full-suite performance. r074 changes only owned native-column4 to row-major copy traversal to16x64 blocking, preserves arithmetic/dtype/bounds/finite checks. Same compiler/script and immutable Matrix include; inherited indentation warnings in unchanged include retained. SDK tests2 pass including rows1/3/129/513/615, N32/1024/3584, partial-row trailing sentinel, wrong/unprepared rows rejection. Three actual SDK7-pair AB/BA component medians candidate/old0.43186/0.32741/0.57384, outputs bitwise equal. No model speed claim yet. r074 model pairing now switches only copy function on the same owned ABI; preserved warmup then3 AB/BA pairs, all responses and inference/selection counter deltas must agree. No test module imported into the live model owner; only the dedicated preflight without an Owner imports SDK tests.

## r074：读取布局配对完成，完整回归运行中

- `execute.py --iteration r074 --case copy-layout-pairing --variant full-batch-sparse-band25 --budget120`：同Owner、仅切换读取函数，3组AB/BA全部完成。完整响应（含confidence）与r072逐项相同，调用、选点与改变数相同；旧中位32.570236秒、新31.083317秒，配对中位比0.957164。预热33.064833秒保留，不挑最快结果；30秒目标仍失败。后端True，关闭0，原服务健康，控制器退出0。
- `execute.py --iteration r074 --case coverage-0000 --variant full-batch-sparse-band25 --budget600`：完整18题回归运行中。目的预先记录，要求对r072完整响应逐项相同并通过原CPU门槛；120秒速度目标不变。
- 下一项扫描策略本机准备：`test_medium_scan_restore_candidate.py`2项、`test_medium_precision_candidate.py`3项通过，语法检查与`git diff --check`通过；未写板端通过。`protection`仅补齐遗漏枚举，实际保护配对尚待运行。

- r074完整回归已核验：271.362672秒，完整响应含confidence对r072差0；原CPU非confidence差0.00899870409、choice/noul一致、字段11/18/题组0同CPU，`home-appliance-check.json` passed。全部20,051token，后端True，关闭0、原服务健康，控制器退出0；120秒速度仍失败。`complete-copy-layout-review.json`保留。检查脚本首次误用通配符匹配到r012 CPU诊断文件（缺records）而报错；改用原协议精确r012文件路径后通过，未改变checker或参照输出。
- r075 `execute.py --iteration r075 --case diagnostic-home-pair --variant full-batch-sparse-native-scan --budget120` 已启动；只在原中长度作用域保留既有native扫描，CPU注意力不改。没有新编译或历史依赖变动，结果待返回。

- r075完整两问题27.058242秒，实际6次native扫描、CPU注意力保持；75项ARM隔离检查通过。对原True CPU最大非confidence差0.125229996且choice改变，原0.01数值门槛FAIL，尽管30秒速度PASS也拒绝采用；不执行其完整18题。错误/回退/OOM0，后端True，关闭0、原服务健康，控制器退出0。`canonical-pair-check.json`保留失败证据。
- r076回归快照重新从已合格r074建立，扫描仍CPU，未携带r075新策略；仅带补齐的protection入口枚举，原34例载荷及本机参照取冻结r011清单，标签/参照不传板端。板端回归待执行。

## r076 / r077：扩展回归发现图片后端状态泄漏

- r076 `--case regression34 --variant full-batch-sparse-band25 --budget120`：34/34输出通过既有原current/previous双数值0.01及choice/noul门槛，保护形状native身份门槛也通过；但主进程最终MKLDNN=False，环境隔离FAIL。原运行时图片`_linear_patch_embed_forward`直接BF16 F.linear触发未支持ARM MKLDNN fallback并改变全局状态；不是隔离子进程检查污染。73项检查通过，关闭0、原服务健康、控制器退出0。`regression-review.json`明确passed:false，未部署。
- r077候选仅把此精确已知CPU BF16 Conv3d线性化forward放入局部MKLDNN disabled作用域，计算仍原函数，异常恢复标志，不修改pin模型文件。新两项ARM测试检查原BLAS逐项相同、True/False恢复、异常与未知目标拒绝；实际结果待返回。`regression34`新增每请求前/后状态证据，不把单个最终True掩盖过程变化。
- 本机`test_execute.py`4、`test_host_run.py`3通过，变更Python语法及`git diff --check`通过。本机无Torch/ARM SDK，真实视觉组件检查在隔离ARM子进程执行，未以mock替代。

- r077结果：75项ARM隔离检查通过，每请求前后MKLDNN均True、视觉scope1次；但34例仅30例满足全部原门槛。image对current非confidence差0.01308144>0.01，对native0.01126172；3条615身份差3.99573e-6/1.03795e-5/1.18696e-5>1e-6，choice/noul均相同。`regression-review.json`明确failed，关闭0、原服务健康，不跑其完整18题。组件与局部状态恢复不等于整次图片推理语义相同。
- r078新候选`request_backend_scope_candidate`保留原公开请求内部的fallback，finally只恢复进入时的MKLDNN enabled值（True或False）；串行Owner限定。2项隔离检查覆盖原请求内False状态保留、payload/返回身份、True/False及异常恢复。不是强制全局True，也不改模型源码。
- `visual-transition-protection-audit`同Owner运行图片→B2→615/115；随后3条问题615用例在True/False分别原native与candidate对照，输出/计数/状态留档；最后原5组AB/BA保护。只作参照与连续调用诊断，不用历史False输出代替原True参照，不放宽原身份门槛。运行中，尚无成绩。

- r078实际75项ARM隔离检查通过，图片→B2→615/115连续调用数值/选择及每请求True恢复通过；图片对current非confidence差0.001819726，B2对r061差9.7483e-6，两个保护形状对native差0。每请求scope只恢复了图片的1次泄漏。
- 3条问题615各4次同Owner后端/模式交叉：True与False各自candidate对实际原native均差0；False原native对历史差0，True原native对历史分别差3.99573e-6/1.03795e-5/1.18696e-5。静态旧参照保留，原门槛不改；完整34例未重新取得通过。
- 原115/615各5组AB/BA：完整数值含confidence差0，中位候选/基线比0.996217/0.989756，原≤1.1保护通过。主后端True，句柄关闭0、OOMFalse、原容器健康，控制器退出0。`transition-backend-review.json`明确范围。
- 同r078源码已启动一次原完整18题回归，目的提前记录，要求对r074完整响应差0并过原CPU数值/原生指标；600秒仅完整性正确性预算，120秒速度目标保留。结果待返回。

- r078最终完整18题已完成并核验：270.821649秒，完整响应字典对r074完全相同；原CPU非confidence差0.00899870409、choice/noul一致、字段11/18/题组0同CPU，`home-appliance-check.json` passed。含confidence差0.01989875267单列。原8/8/2及20,051token保留，CPU扫描优化调用增量0；75项隔离检查通过，主后端True/Torch3/highest，request scope1次/恢复0次（纯文本无状态泄漏）。峰值8100982784字节，max/oom/oom_kill0，句柄关闭0、原服务健康、控制器退出0。原120秒速度FAIL，完整34例与全样本未验收，未部署。
- 最终命令：`execute.py --iteration r078 --case coverage-0000 --variant full-batch-sparse-request-scope --budget600`；venv Python调用`check_home_appliance.check(r078完整result, r018完整CPUresult, r012原失败result)`，再断言整个响应字典等于r074、运行时状态/75项检查/CPU扫描增量/内存事件。审核脚本保存在忽略目录r078的`review_full.py`、`review_transition.py`，审核JSON独立保留。
- 本轮收尾：无控制器或后续板端任务，原服务恢复健康，定时任务仍停用。下一轮先固定34例独立参照上下文，不把现有局部通过写成完整发布资格。

## r079：固定后端上下文完整34例（2026-10-20）

用户要求全部34例过原数值门槛后才提速。新增`context-regression34`完整运行入口：原CPU/reference、原native/baseline、r078候选分别新建独立Owner，前两组不加载新数值适配器，三组相同公开请求结束恢复策略。每例记录原载荷身份、状态、调用计数和完整输出；前/后必须MKLDNN True、Torch3/highest/autocastCPU False，CPU NPU/graph增量必须0，图片原内部fallback保持。原输入SHA和34条顺序不变。

原CPU已启动：`caffeinate -i output/decision-index-audit/venv/bin/python official-long-input-experiment/execute.py --iteration r079 --case context-regression34 --variant reference --budget120`。结果待返回，未改性能或宣称过门槛。冻结新参照必须在候选执行前完成，来源只能独立原CPU/native；历史文件保持并双列比较，原`check_regression.check_pair`不改。

本机`test_context_regression.py`2、`test_check_context_regression.py`2、`test_regression_gate.py`3、`test_execute.py`4通过，语法与`git diff --check`通过。新增checker测试首次fixture把native与CPU都设成0.5，导致先触发native门槛而非指定CPU门槛；修正fixture使native0.502、CPU0.5后验证原CPU0.01拒绝，未改门槛实现。覆盖全部34顺序、后端漂移/CPU意外NPU拒绝、参照身份不符/覆盖拒绝、原0.01及保护1e-6失败留档。

- r079首次CPU参照运行由新增检查真实拦截：words-512 CPU增量全0，single却有12次graph调用；原`owner.use(current)`把bank设fixed而非cpu。请求后端均True，运行在第2例后拒绝，完整34未完成，证据与原容器健康保留。该输出不冻结作新CPU参照。
- r080重新建不可变快照，只为此reference入口显式bank.mode=cpu（原已禁用engine eligibility、恢复Linear原FP32 forward），确保原CPU参照的NPU/graph增量真实为0；不改candidate/native路径或旧cpu-full协议。原输入/门槛/模型不变，从34例首条重跑。

- 新上下文检查额外保留历史CPU/native的0.01与离散判定门槛，不能通过更换上下文消除旧原CPU数值失败；历史严格native身份因已证明False上下文而双列报告，新同上下文仍严格1e-6。`test_check_context_regression.py`新增此防绕过检查后3项通过，原门槛函数未改。此规则在候选运行前冻结记录。

- r080独立CPU34例完成：共513.771秒，仅正确性参照，每例NPU/graph增量全0，后端全部True/Torch3/highest，public request scope34次/图片恢复1次，关闭0、原服务健康，控制器退出0。独立native34例完成，同上下文/完整输出/顺序与源身份通过，615 native和115图路径实际命中，关闭0、服务健康、退出0。
- venv Python `check_context_regression.freeze(cpu_folder,native_folder,r080/context-references.json)`通过，参照在候选启动前冻结、来源仅CPU/native，输入SHA/两份完整运行源码与身份相同。原native对纯CPU最大非confidence差0.00943617884（extra-shape115-4），全部34例≤0.01且离散判定一致。独立参照不覆盖历史文件。
- r080独立候选完整34例已启动：`execute.py --iteration r080 --case context-regression34 --variant full-batch-sparse-request-scope --budget120`。数值结果尚待返回，未改性能。

- r080候选34/34通过原门槛：新纯CPU最大非confidence差0.00943617884，含confidence0.01832714332单列，choice/noul/usage/model一致；115/615同上下文native全部数值差0。历史CPU/native的0.01及离散门槛也通过，旧3条False参照身份差单列保留。75项ARM检查通过，每请求前后后端True/Torch3/highest，关闭0、原服务健康、退出0。原check_pair源码SHA841c6b517241d3d9af051bcd18a2f51a83c2d30a4aef4b05eaec1e85f1195683不变。
- 仅在r080 passed后开始r081性能实验：保留相同nm_raw及原bad4有限值/65488范围检查，省去稀疏MLP中被丢弃的BF16转换、存储和输出分配；CPU修复和实际BF16边界不改，保护形状旁路，worker全部join后才读取FP32。原C include、编译脚本不改，原库保留，未并行推理/编译。
- 本机test_native_output_policy.py两项通过：相同prepare/copy/bucket、恢复作用域，以及worker失败仍join另一个worker。r081真实SDK预检4项已完成，检查FP32逐位一致、295/320双down及其他真实行数、原大有限值范围拒绝、未准备/错行数拒绝；最终清理/恢复仍待DONE，尚无整模型提速结论。
- 后续性能候选源文件允许变化，但必须在推理前独立固定完整lab源清单；原模型/依赖身份仍与参考相同。新check_context_regression来源检查不改变check_pair，原门槛源码SHA必须与冻结参照相同。定向checker四项通过，包括新源清单不符和基础身份变化均拒绝。

## r081 rejected; r082 scan operator diagnosis; r083 component preflight

- r081 completed all3 AB/BA pairs. Complete response dictionaries equal qualified r072, inference/repair/selection counters equal,192 discarded outputs actually elided per enabled request. Ratios0.9864222886/0.9802330036/0.9758639531; median0.9802330036. Copy median30.855112s, experimental30.245199s. The predefined requirement was below0.98: REJECTED. No full34/full18 or deployment of this experiment.79 ARM checks passed, lifecycle zero, backend True/Torch3, OOMFalse, old service healthy, controller exit0.
- Rejected overlay sources moved to rejected-native-output-elision/. Active C++/MLP/host source SHA values exactly restored to r080. Active checker exactly restored to qualified SHA42a22dbd5aa10dbd6d5c698286f0bd629d57511ca804640679f3ddba30bac1f6. Complete board snapshots and private results remain preserved.
- r082 scan-operator-profile, original r080 math:6 original CPU scan calls, only1 profiled, full response dictionary equals r072. Repair/MLP errors/fallback zero, backend True/Torch3, handles zero, OOMFalse, service healthy, exit0. Actual callable chunk_blas2.install.<locals>.forward on B2/T1193. Aggregated operator self times: bmm39 calls819.352ms, copy152.295ms, triangular solve127.579ms, mul117.226ms. Profiler wall2.106s and whole request38.676s include instrumentation overhead, NOT latency results; inclusive values overlap, one scan does not represent the full suite.
- r083 scan-bmm-preflight is diagnostic only. Capture actual scan BMM operands with unchanged shape/stride/values; reproduce original3-thread live FP32 output first, then7 alternating3/1-thread component pairs, requiring bitwise equality. Thread switch cost separately recorded. Whole model remains3-thread, original response and final backend must match. Running; no production strategy changed.

- r083 low-level BMM capture failed its required-shape assertion, before any component timing. Full request response still exactly equaled r072, threads restored3,75 ARM checks passed, handles zero and old service healthy; incomplete diagnostic preserved, no speed claim.
- r084 corrected the diagnostic to intercept the actual high-dimensional Tensor matmul call used by CPU scan. Original3-thread live output reproduced bitwise; actual5D and4D input values/strides retained without flattening the replay. Seven alternating3/1-thread pairs per class remained FP32 bitwise equal. Component ratios1/3:0.9410713311 for batch class320 and0.9617872353 for class32. Median thread-switch cost about21microseconds, separately measured. Full model response equals r072, backend restoredTrue/3/highest, lifecycle zero, OOMFalse, service healthy, exit0. Component results are NOT an engine speed result.
- After this evidence, r085 introduced only a scoped1-thread original CPU scan in the existing medium precision domain; it restores3 after success/error, preserves short protected bypasses and leaves repair/reference settings untouched. Two ARM scope/failure tests added. Whole-model3-pair experiment then stopped due an unrelated historical summarize module being resolved from sys.path. The new-policy warmup35.8837s and stock paired request30.8175s were both saved and exactly equal to qualifiedr072; no valid paired speed conclusion.77 checks passed, exit restored service; interrupted evidence remains.
- Fixed the diagnostic import using an explicit local summarize.py file path and unique module name; one focused name-collision regression test passes. r086 is a fresh independent3-pair run, same scan policy, unchanged frozenCPU/native references, purpose and source map pinned before launch. Full34/full18 still required if numerical and predefined2pct speed gates pass. No scan thread policy has been adopted or deployed.


## r086 final review and active-source restoration

- Command: `execute.py --iteration r086 --case scan-thread-pairing --variant full-batch-sparse-scan-threads1 --budget120`. All3 AB/BA pairs completed; ratios1.0796706814/1.1107780619/1.0850748753, median1.0850748753. Original3-thread median31.471618s, scoped1-thread34.212711s: about8.51pct slower, REJECTED. No further34/full18 or deployment of this policy.
- Saved warmup plus all6 responses exactly equal qualifiedr072. Against independent originalCPU r061, max numeric delta9.74830259e-6, choices/noul/usage/model equal. Every enabled request actually has6 scopes, disabled0.77 ARM checks passed; final backendTrue/Torch3/highest, lifecycle zero, OOMFalse, cleanup errors empty, retained service healthy, controller exit0. Private `r086/pair-review.json` and immutable source pin preserved.
- Experimental scan policy and harness archived in `rejected-scan-threads1/`. Active probe restored from r084 diagnostic-only snapshot. Active C++/MLP/host/context/request-scope source hashes match qualifiedr080; checker SHA42a22dbd5aa10dbd6d5c698286f0bd629d57511ca804640679f3ddba30bac1f6 unchanged. Evidence: `r086/active-source-restoration.json`. Rejected output-elision overlay remains archived separately.
- Post-cleanup SSH docker inventory: only decision3-edge-graph running, healthy, bound192.168.1.164:8900. No new service deployment; full1000 evaluation and automation remain paused.

- Final local verification: `python3 -m unittest discover -s official-long-input-experiment -p test_context_regression.py -v`2/2; `test_check_context_regression.py`3/3; `test_regression_gate.py`3/3; `test_execute.py`4/4; `test_host_run.py`3/3. Total15 focused checks pass. `python3 -m py_compile official-long-input-experiment/*.py` and `git diff --check` pass.
- Independent offline re-audit used qualified read_run on all three r080 folders, unchanged check_pair on all34 responses, historical CPU/native0.01 and all discrete gates, reference hashes and protected all-numeric delta0. All passed. Existing result files were not overwritten. No controller process remained after cleanup.


## r087: complete fixed-request cost ledger (diagnosis only)

- User requested a complete cost ledger for the approximately31s B2/T1193 request before selecting further optimization. No compute policy, model, batch, input, numerical gate or historical dependency is changed. An isolated r087 snapshot was cloned from qualified math plus r084 diagnostics. Purpose/source maps were pinned before the actual launch.
- New instrumentation partitions owner-thread leaf intervals across prepare/respond, decoder, scan, attention, projection, MLP, sparse repair, explicit FP32 reads, native calls and joins. Concurrent worker/SDK intervals are separate overlapping views. Public synchronous SDK entry-to-return is intercepted by a diagnostic pass-through library using the pinned public header and unchanged runtime; it includes driver/submit/wait, not pure device time. Only the request-cost-ledger case preloads it. Ordinary regression and service launch environments remain unchanged.
- All6 real CPU scans are instrumented for high-dimensional Tensor matmul, triangular solve and explicit conversion, without changing arguments, strides, arithmetic or outputs.3 plain/profile AB/BA pairs follow one preserved shape warmup. Full response dictionaries and inference/selection counts must match; originalCPU0.01 and discrete decisions remain required offline. Wall closure must be0ns; overhead and all observations retained.
- Local `test_request_cost_ledger.py`:3 pass (nested concurrent union, wall closure, original Future return/error and descriptor restoration, malformed interval rejection). `test_host_run.py`:4 pass, including preload isolation. Changed Python syntax and git diff --check pass. Diagnostic SDK library compiled on ARM with g++ and public SDK header; model tests/results pending.
- One prelaunch CLI invocation was rejected by argparse for --budget120; corrected to --budget 120 before any service change or model request. Initial pin and correction retained separately; final pin adds all-scan operator subspans. No invalid timings were included.

- r087 actual launch: `caffeinate -i output/decision-index-audit/venv/bin/python official-long-input-experiment/execute.py --iteration r087 --case request-cost-ledger --variant full-batch-sparse-request-scope --budget 120`. Preserved shape warmup32.264125s; plain31.014209/30.989520/30.878245s, profile31.811284/31.337204/31.388456s. Plain median30.989520s, mean30.960658s; profile mean31.512314s; paired median overhead ratio1.016523. All complete responses equal qualifiedr072, every inference/selection counter delta equal.75 ARM isolated checks pass; backendTrue/Torch3/highest/autocastFalse before/after each request, handles0, OOMFalse, cleanup errors empty, retained service healthy, controller exit0.
- Actual payload SHAacecf821f479ac8a7af688d9bcfce7affd4bf578c72a3e70f0271cce361603bb matches originalr018 input. Lengths1193/1102, effective2295, padded2386. Initial purpose text mislabeled padded2386 as effective; original purpose preserved, geometry clarification recorded. No payload or batching changed.
- `output/decision-index-audit/venv/bin/python official-long-input-experiment/build_request_cost_report.py output/decision-index-audit/iterations/r087/20261020T040623-request-cost-ledger-full-batch-sparse-request-scope-plain` passes: source pin matches, only probe.py/host_run.py differ from common qualifiedr080 files, unchanged original numerical gate sourceSHA, originalCPU max nonconfidence9.74830259e-6, discrete decisions/usage/model equal. All3 full27-category wall ledgers independently reconstructed with exact0ns closure. Per profile6 scans,234 matmuls,12 triangular solves,6 attention calls,372 SDK calls with0 statuses. Raw events and audit remain private.
- SDK per-thread duration sum mean3.189228s, concurrent wall union1.791978s; cannot add these or call either pure device execution. Profile process CPU mean71.386048 CPU seconds is separate from wall time. Normalized-to-plain conditional estimates: repair/read11.630718s, scan7.493383s, attention4.997648s, together77.91pct. SDK-overlapped call/join1.760594s. These normalized values and25/50pct reduction scenarios are estimates, not newly achieved gains.
- Reports generated as REQUEST_COST.md and REQUEST_COST.html with aggregate values only, no questions/answers/labels. Browser screenshot visually checked: five-part bar, readable tables,27-item details collapsed, explicit estimate and SDK caveats. Local preview serves only this experiment directory on127.0.0.1:43993. No RK inference/controller remains; original graph service healthy on8900. No optimization adopted or deployment performed in this diagnostic round.


## r088: bounded selected-dot ILP experiment, before performance adoption

- User authorizes selected-dot optimization only: same mask, K accumulation order, numerical gates; require more than1s whole-request saving before full34/full18. Round purpose pinned at04:15:47UTC with45min budget and only widths8/16. Component screen uses r087 normalized selected-dot cost5.679774793s; predicts savings only, never model performance. Whole-model gate is median of three paired old-minus-new times above1s, with all observations retained.
- Own selected_dot_ilp.c interleaves independent outputs only: one accumulator per output lane, explicit K0..K-1 FMA order, no partial K reductions, original final +0 then BF16 tie-to-even. Width4 is an original-order FP32 debug control, not an extra speed candidate. CPU3/OMP3 and no-fast-math flags preserved, scratch bounded to tiny local arrays. Original sparse_cpu.c/library and all model/SDK libraries unchanged.
- Three new ARM checks cover original BF16 output and4/8/16 FP32 bit parity across K4/64/1024/2048/3584/6144/10240, noncontiguous row strides, row gaps and odd tails; empty/duplicate positions, invalid index/order; nonfinite and wrong-backend rejection. Tests run only in the isolated child with scoped backend state.
- Real preflight keeps old model output: every actual selected-dot call is replayed on private buffers for width4/8/16, selected outputs first poisonedNaN, whole BF16 output equality and FP32 bit equality required. Up to3 alternating component pairs per width/geometric class; real positions hashed and preserved, no labels used. These timings are diagnostic only. Sources and compiler manifest pinned before r088 launch.
- Commands: `bash /opt/decision3-official-opt-20261019-r088/build_selected_dot_ilp.sh` succeeded with gcc; local Python syntax and git diff --check pass. `execute.py --iteration r088 --case selected-dot-preflight --variant full-batch-sparse-request-scope --budget 120` running; ARM test/model results pending.

- r088 completed and STOPPED at component stage.78 isolated ARM checks passed (75 inherited plus3 selected-dot checks). All372 real calls /8,063,030 positions /9 geometries passed whole BF16 output comparison with installed original kernel and selected FP32 bit parity across4/8/16 original-order chains. Full old-model response equals qualifiedr072; originalCPU max nonconfidence9.74830259e-6, discrete choices/noul/usage/model unchanged. Main backendTrue/Torch3/highest/autocastFalse, lifecycle0, OOMFalse, retained service healthy, controller exit0.
- Three AB/BA component pairs per geometry/width, weighted by actual selected counts: width8 ratio1.1422175819 (about14.22pct slower), width16 ratio1.2573630728 (about25.74pct slower). Based on r087 selected-dot5.679774793s, conditional estimates are negative savings -0.807764/-1.461764s. A single615x1024-to3584 geometry for width8 is about5pct faster, not a whole-workload gain; not cherry-picked. Neither width meets the predeclared1s potential gate. NO candidate whole-request pairing, NO full34/full18, NO adoption/deployment or full1000 restart.
- Diagnostic full request61.399170s includes all private replays and component pairs; it is NOT candidate latency. Core reason for slowdown not established; no unverified register/bandwidth explanation asserted. The preflight whole response is the old model, not a candidate quality acceptance.
- Review command: `output/decision-index-audit/venv/bin/python output/decision-index-audit/iterations/r088/review_preflight.py` passed. It checks source pin, exact call/selection totals against r087, every variant result, originalCPU gate, backend/lifecycle/cleanup and weighted screen. Evidence: r088/preflight-review.json and immutable raw run folder20261020T041926-selected-dot-preflight-full-batch-sparse-request-scope-plain.
- Overlay sources archived in rejected-selected-dot-ilp/, including the exact tested probe and new3-test source. Active probe restored byte-for-byte from r087; active sparse CPU, backend, enabled sequential repair, rounding guard, fused MLP, C++ and original gate source hashes match r087. r088/active-source-restoration.json preserves verification. Original model and math libraries never rebuilt. Public archive summary has aggregates only.
- Documentation and existing cost report now record the rejected first attempt; r087 cost audit/regeneration still passes. Local syntax and git diff --check pass. Original service remains the qualified retained graph deployment; this round did not improve or replace it.

## r089/r090: selected-dot slowdown diagnosis, frozen binaries, no optimization adopted

- Scope/purpose pinned before each launch:45min total diagnosis budget; fixed original/4/8/16 r088 ELF, no recompilation, same real selected positions, BF16 values, per-output K FMA order and original gates. Frozen math libraries and active numerical gate unchanged. CPU4–7 is Cortex-A72 (MIDR part0xd08), PMU type9; no A76 assumption.
- No perf executable installed. Root-host perf_event_open access independently verified. Host-only observer authenticates Unix peer UID/PID/owned diagnostic container ancestry, then attaches thread-specific PMU groups. Observer CPU0–3, target CPU4–7; unchanged Docker caps/seccomp/sysctl.6events/group with >=99% running-fraction gate, no scaling of multiplexed samples. Target thread sets must remain stable. Both observers printed PMU_CLOSED and exited0, socket removed.
- r089 command: `caffeinate -i python3 official-long-input-experiment/execute.py --iteration r089 --case selected-dot-diagnosis --variant full-batch-sparse-request-scope --budget 120`. Run20261020T051005-selected-dot-diagnosis-full-batch-sparse-request-scope-plain. Original model capture33.269226s includes extra capture work, not candidate latency.372 calls/8,063,030 selected positions match all r088 call-index/shape/count/SHA fingerprints.9representative first-per-geometry captures, original/4/8/16 replay,3event suites,4rounds,11workloads including two pad64 counterfactuals:528 windows.80 isolated ARM checks pass. All replay BF16 bits identical, min running fraction1.0, no thread changes.
- r089 audit initially exposed an invalid metadata assumption that every weight base was64B aligned:5geometries actually offset16B. Input copies preserve original0B alignment; original weights remain untouched. First pad64 control changed alignment16→0 as well as stride, explicitly retained as confounded. Metadata check now records actual0/16B, numerical gates unchanged. This motivated a separate preregistered factorial control, not width/parameter searching.
- r090 command: `caffeinate -i python3 official-long-input-experiment/execute.py --iteration r090 --case selected-dot-diagnosis-control --variant full-batch-sparse-request-scope --budget 120`. Run20261020T051952-selected-dot-diagnosis-control-full-batch-sparse-request-scope-plain. No full model setup/inference: same SHA-pinned private r089 captures10/15,2shapes×2strides×2base alignments×4frozen kernels×2PMU suites×8balanced rounds=512windows. Every BF16 value bit-equal, running fraction1.0, stable thread sets. Full raw slow values retained.
- Same alignment16B,512x2048-to1024 width16: core-group median43.665997→19.701783ms, paired median ratio.453145 with8/8 rounds faster; L1 refill248.9188→69.6108/selected, retired instruction4529.9726→4523.8120/selected. Independent cache-group paired ratio.469387,8/8 faster; L2 refill55.2956→55.5516/selected while L2 accesses892.194→302.172. Both base alignments show stride sensitivity; changing only alignment does not remove slowdown. Evidence supports stride-related L1 locality/conflict/multi-stream contention, not proven DRAM bandwidth saturation or physical-address-level proof of every set conflict.
- CPU endpoint frequency always2208000kHz. DMC auto-frequency observed534000000/1320000000/1968000000/2736000000Hz, left unchanged and explicitly reported. This prevents interpreting observed ms or percentages as a fixed-DDR-clock causal decomposition or whole-request gain. No stall-class events: remaining issue/rename/load-use scheduling costs not uniquely attributed.
- Frozen objdump K hot loops: original190c–1984 (31instructions/5loads/4FMLA); new4 940–9b8(31/5/4),8 e20–f00(57/9/8),16 1740–1900(113/17/16). No SP-based memory operand in any hot loop. Width16 post-K output-temporary stack stores identified separately. This rules out repeated K-loop spilling, not every possible register-pressure effect.
- Read-only `audit_selected_tail.py` parses only owned Torch ZIP tensor metadata and selected int64 storage, SHA verified, restricted pickle classes/no Torch import/no inference.9actual row-count distributions:16-way extra computed lanes9.26%–25.17%, original4-way2.05%–5.51%.512x2048-to1024:4way3.171%,16way16.178%; hot-loop instruction prediction ratio1.026169, PMU retired-instruction actual ratio~1.024517. Tail duplication partly reverses static per-output instruction savings.
- Audit commands passed: `python3 official-long-input-experiment/review_selected_dot_diagnosis.py output/decision-index-audit/iterations/r089/20261020T051005-selected-dot-diagnosis-full-batch-sparse-request-scope-plain`; `python3 official-long-input-experiment/review_selected_dot_cache_control.py output/decision-index-audit/iterations/r090/20261020T051952-selected-dot-diagnosis-control-full-batch-sparse-request-scope-plain`. r089 original response equals qualifiedr072, CPU nonconfidence max9.74830259e-6, choices/noul/usage/model equal; backendTrue/Torch3/highest/autocastFalse, handles0. Both controllers exit0, no OOM/budget/cleanup error, identical retained container restoredhealthy. Private evidence under iterations/r089/r090; public aggregates and exact overlays archived selected-dot-diagnosis/.
- Focused local checks pass: `PYTHONPATH=official-evaluation python3 -m unittest discover -s official-long-input-experiment/selected-dot-diagnosis/r090 -p test_host_run.py -v` (5); `python3 -m unittest discover -s official-long-input-experiment/selected-dot-diagnosis -p 'test_dot_pmu_monitor*.py' -v` (5, including unrelated-peer rejection); restored active `test_host_run.py` (4, pre-existing SDK isolation retained). Python syntax and git diff --check pass. Report generator `build_selected_dot_diagnosis_report.py` uses audited aggregate JSON only; existing r087 cost report regeneration/audit still passes and links new diagnosis.
- Active probe/host and checked numerical core files restored by SHA to r087; active host tests restore pre-diagnostic4tests, including local post-r087 SDK isolation test. Restoration evidence r090/active-source-restoration.json. NO new kernel/layout adopted, NO new candidate whole-model timing/34/18/full1000/deployment; original sample, gates,25prompts and .idea untouched. No automation created or resumed.

## r091/r092: bounded cached row layout and hybrid tails, rejected before model pairing

- User authorizes fixed cache-friendly weight layout plus tail handling, same positions/K order/numerical gates, accounting for preparation/memory; full34/full18 only after real complete-request saving>1s. r091 purpose pinned before launch adds a conservative component-potential screen>1s before model pairs;60min bounded round, no timer/full1000/deploy. Same checkpoint, input payload and originalB2/T1193,615/115 warmups, CPU4–7/Torch3/8GiB.
- New separate selected_layout_tail.c/libselected_layout_tail.so only. `taskset -c 0-3 bash /opt/decision3-official-opt-20261019-r091/build_selected_layout_tail.sh` succeeded, gcc -O3 -ffp-contract=off -fopenmp -std=c11 -march=armv8-a+simd. Old C/CPP/native libraries SHA equal r089. BF16 row copy strideK+32,64B, original module weight unchanged; explicit version/identity guard, private versionless weights allowed only original guarded owner packed bank, request-scoped64MiB LRU evict-before-allocate, finally release on success/error. Full16/full8/original4-form tails preserve K FMA/+0/BF16, <=3unused lanes per row.
- New focused ARM checks: mixed strides and tail counts acrossK4/64/1024/2048/3584/6144/10240; old BF16 bits and continuous-orderFP32 bits; empty/repeated/invalid/unsorted positions, wrong backend/versionless guard; cache reuse/version mutation/weight replacement/eviction/oversize; request exception release and owner restore. r09179checks pass,0fail/error/skip; r092 adds pure-original4 layout check for80checks. Isolated child MKLDNNFalse remains child-only, parentTrue before/after/run cleanup, no numerical gate relaxation.
- r091 actual: `caffeinate -i python3 official-long-input-experiment/execute.py --iteration r091 --case selected-layout-preflight --variant full-batch-sparse-request-scope --budget 120`; run20261020T133445-selected-layout-preflight-full-batch-sparse-request-scope-plain. Three fixed mechanisms: new4helper+layout,hybrid16/8/4+original layout,hybrid+layout.372real calls/8,063,030 selected positions, all fingerprints match r088; each BF16 bit pattern equals original, FP32bits equal qualified original-order4debug.9geometries×3AB/BA pairs, weighted by actual selected counts. Instrumented old-model68.126755s is NOT candidate latency.
- Review found control confound: r091 label layout_only still calls new compiled4helper; tail_only includes wider main groups and tail dispatch. Numeric parity passes but cannot isolate pure layout/tail causal performance. Retain all r091 data/source/purpose; r091/control-confound-review.json records assumption/evidence/correction. No implementation parameters swept.
- r092 actual: `caffeinate -i python3 official-long-input-experiment/execute.py --iteration r092 --case selected-layout-original4-preflight --variant full-batch-sparse-request-scope --budget 120`; run20261020T134215-selected-layout-original4-preflight-full-batch-sparse-request-scope-plain. Identical native sources/binaries as r091, no rebuild. OriginalLayout uses unchanged EnabledSequentialRepair/d3_sparse_repair_diagonal qblock16384 on the same padded-cache view; private debug ref cannot overwrite original result. Same372/8,063,030 fingerprints and bit tests,9geometries3AB/BA. Instrumented old-model53.942277s is NOT candidate latency.
- Results: r091 combined weightedratio.9392994267; observed prepare+release.2550642723s; conditional estimate5.679774793*(1-ratio)-cost=.0897013140s. New4helper+layout ratio.8805427642/cost.2589055528/estimate.4195846440s. Hybrid+original layout ratio1.4234170385/negative estimate-2.4049189843s. r092 pure original4+layout ratio.8770337472/cost.2474248575/estimate.4509957657s; gross estimate without any preparation only.6984206232s. No potential above1s, both primary gatesfalse. These are representative component extrapolations, not actual end-to-end latency; initialization/context/scheduling not fully counted until model pairs, which were NOT run.
- Each padded implementation sees72unique original modules totaling456MiB BF16;72builds,327hits (300actual reuses+27benchmark calls),62evictions,0invalidations,0oversize bypasses. Live payload peak66,748,416B=63.65625MiB; cumulative allocation490,340,352B=467.625MiB, not a peak. Each module copied onlyonce: increasing cache would not reduce B2 copy volume. Caches andentries0after end. r091 simultaneously keeps2pad caches, r092one; cgroup peaks6,453,899,264/6,455,914,496B are whole-run high-water including setup/warmup/checks, not candidate-specific delta; RSS/shared-memory counters not additive. No claim RSS immediatelyreturns after tensor release.
- New library objdump: compute4 Kloop808–878 has29instructions; compute8 8e0–9b0 has53; compute16 a44–c04 has113. No SP-based memory access in any Kloop; compute16 ABI callee saves occur outsideK. Full assembly/private parsed review in r091; public assembly-summary.json. No new PMU measurement or unique attribution of remaining scheduling/cache costs.
- Offline audits passed: `python3 official-long-input-experiment/review_selected_layout_preflight.py output/decision-index-audit/iterations/r091/20261020T133445-selected-layout-preflight-full-batch-sparse-request-scope-plain`; same for r092/20261020T134215-selected-layout-original4-preflight-full-batch-sparse-request-scope-plain. Source pins/native identity, all372fingerprints/modes/bits, every pair, exact counts/cache bounds, parentbackend/lifecycle/cleanup, original response equality and CPU gate checked. Original responses equal qualifiedr072; CPU nonconfidence9.74830259e-6,choice/noul/usage/model equal. Actual new candidate public-request inference not completed, so no full candidate accuracy or speed claim.
- Both controllers DONE completedTrue/service_healthyTrue/exit0;0OOM/budget/cleanup errors; all original handles0, same retained container restored. Selected adapters/copies stay private; active probe and checked math files restored by SHA. All candidate source overlays archived rejected-selected-layout-tail/r091,r092 with common C/build and public aggregate summaries (including actual geometry weights and all component pairs, no labels/tensors). No candidate adopted, no modelpair/full34/full18/full1000/deployment or automation. `.idea` and25prompts untouched.
- Local verification: Python compileall for archive and py_compile for reviewer/report builders; bash -n archived buildscript; `git diff --check`; aggregate report rebuild and all local HTML links checked. Graphical report SELECTED_LAYOUT_TRIAL.html and Markdown generated from audited summaries, existing diagnosis links to outcome. ARM unit execution above is the native/runtime verification; no unsupported local ARM execution substituted.

## r093: original four-lane layout, three complete cold-cache request pairs

- User explicitly requested actual3full pairs despite prior component estimate. Protocol written before launch in selected-layout-request-pairs/README.md: original qualified4-lane ELF + pad64B/row,64MiB requestLRU; sameOwner/model/input/batch/padding/selection/Korder/backend/gates; old/new,new/old,old/new after old/new full shape warmups. Everynew request cold, preparation/eviction/release inside full system_one wall; full one-off import/install conservatively deducted from each pair. No16/8candidate, parameter tuning, compilation, SDK interposer or component replay in timings.
- Command: `caffeinate -i python3 official-long-input-experiment/execute.py --iteration r093 --case original-layout-pairs --variant full-batch-sparse-request-scope --budget 120`. Actualrun20261020T152924-original-layout-pairs-full-batch-sparse-request-scope-plain.81isolated ARM tests pass (75qualified +5inherited layout +1new originalScope success/exception/toggle/restore). All inherited r092 files except probe byte-identical before launch; immutable /lab source snapshot verified offline. Original libsparse_cpu.so SHA17da1be76215bdf3ffd58548205a44c453d3a4b911655254c22b896237e216a0, not rebuilt.
- Complete shape warmups old33.659143173/new30.392586346s, separately retained.3timed pairs old/new31.325267610/30.842188156,30.687463353/30.750239684,30.624519210/31.817207595s. One-off installation0.013850890s. Paired net savings+0.469228564/−0.076627221/−1.206539275s, median−0.076627221s. Strict>1s fails; no34/full18/full1000/deployment. Slow thirdnew retained. Separately computed old/new medians30.687463353/30.842188156s must not be substituted for paired-difference median.
- All8complete warm/timed responses equal qualifiedr072; original pureCPU nonconfidence9.748302590040225e-6<=.01,choice/noul/usage/model unchanged. Allcall/selection/modification deltas identical:372repairs/8,063,030selected,0density rejection/readerrors/CPUfallback. Backend before/after everyrequest True/Torch3/highest/autocastFalse; no numerical gate change (check_regression.py SHA841c6b517241d3d9af051bcd18a2f51a83c2d30a4aef4b05eaec1e85f1195683). This is candidate B2qualification, not a newly executed full34 or official accuracy.
- Pernew request:72builds,300hits,62evictions,0invalidations/oversize; exact livepayload peak66,748,416B=63.65625MiB <=64MiB; cumulative allocation490,340,352B=467.625MiB is traffic, not livepeak. Preparation/eviction times0.260408270/0.255011838/0.276641079s; release0.002386946/0.007108920/0.003143618s. All already within full wall. Combined prepare+release median0.262795216s. Cache cleared after each success; independent exception test passes. Whole cgroup peak6,766,145,536B includes model/setup/tests/warmups and is not candidate-only increment or additive withRSS. Perrequest snapshots published. NoOOM/cgroupmaxevent.
- Offline audit command: `python3 official-long-input-experiment/review_original_layout_pairs.py output/decision-index-audit/iterations/r093/20261020T152924-original-layout-pairs-full-batch-sparse-request-scope-plain`. PASS numerical/runtime/sourcepins/81checks/complete-response/counter/cache/backend/lifecycle/controller; speed_gate_passedFalse. Public summary stripped question/label/response/operating-state payloads.
- Controller DONE completedTrue/service_healthyTrue,exit0,cleanup errors[],allhandles0; candidate method/kernel restored andcache empty. Same original container07577e5f9827249b3803dacef6b9b78048a008585fce285d5ec2c85cd2dade24 runninghealthy. Direct published-IP HTTP health ready, graph-half-v1 retained. Host-loopback curl initially refused because8900 binds192.168.1.164 only; inspected binding and container healthcheck, then checked correct published URL successfully. No service config change. No owned worker remains. Finalservice-health evidence saved ignored r093folder.
- Local checks: AST/whitespace9Python files;10active core hashes equal frozenr093 inherited source; active probe SHA3be3b69c86dea66770d3ae3f9f29580222edda2eabb964fdf7ed34931758bdea unchanged. Newadapter/tests inherited byte-identicalr092. `git diff --check` passes. `python3 official-long-input-experiment/build_selected_layout_pairs_report.py` and `python3 official-long-input-experiment/build_selected_layout_trial_report.py` rebuild result HTML/Markdown; all local HTML links resolve and newIAB result page renders audited3pairs/costs/gates. Old component page links toactualfollowup while retaining historical estimates. No secrets/raw labels/model outputs in public summary; .idea/25prompts untouched. No automation/fullsample resumed.
