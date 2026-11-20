# 未采纳：请求内权重布局缓存与16/8/4尾部分派

2026-10-20，r091/r092。[结果图表](../SELECTED_LAYOUT_TRIAL.html) · [可编辑正文](../SELECTED_LAYOUT_TRIAL.md)。实现已完成并通过私有组件位模式核验，但没有满足预先记录的继续门槛，**停止组件阶段，不进入候选整请求配对、完整34例、18题或部署**。

## 预先固定的实现

- 同一BF16权重值和逻辑维度；派生副本每行增加32个BF16元素（64B），不修改原模块weight/原NPU权重，也不改变选点、2.5e-7经验带、10%/65,536预算、输入批次或模型12层。
- 请求内LRU缓存的活跃派生存储上限64MiB，先逐出再分配，每请求从空缓存开始，正常/异常结束清空。保留权重对象/形状/stride/地址/version守卫；无version的私有打包权重仅在原Owner packed bank身份守卫下使用，不跨请求保留。
- 满16点用16路，剩余满8点用8路，再交4路处理；不足4最多补3点。每输出K0..K-1的FP32 FMA链、最终+0、BF16舍入不改，不使用fast-math或分段归约。
- 只编译独立`libselected_layout_tail.so`，原`libsparse_cpu.so`及其他数值核心不重编译。固定gcc `-O3 -ffp-contract=off -fopenmp -std=c11 -march=armv8-a+simd`。64MiB是受8GiB整机容器与旧完整18题内存余量约束的活跃存储预算，不是调参搜索结果。

## r091与对照纠正

r091比较三个固定实现：新4路helper＋布局、16/8/4内核＋原布局、16/8/4内核＋布局。79项ARM隔离检查通过；真实372次/8,063,030选点、9几何全部BF16位模式与原内核相同，FP32位模式与已核验的连续顺序4路调试参照相同。每几何3组AB/BA组件计时，按该几何实际总选点量加权。

审核发现：第一项虽然名为`layout_only`，仍用了新编译4路helper，混入代码生成/调用方式差异；第二项`tail_only`也同时改变宽主循环，不能把它当作仅改变尾部的因果对照。**旧源与结果保留，不能据其退化断言尾部处理本身慢。**

r092新增`OriginalLayout`，保留完全相同的缓存、步幅和预算，调用原合格`EnabledSequentialRepair`/`d3_sparse_repair_diagonal`（qblock16384）执行，单独分离布局效果；没有重编译任何库，没有再调宽度/行间隔/容量/线程。80项ARM检查和同样372次真实私有位模式核验通过。

| 实现 | 加权组件新/旧耗时比 | 准备＋释放 | 扣已测成本后的整请求条件估算 |
| --- | ---: | ---: | ---: |
| 原4路＋布局（r092） | 0.877034 | 0.247425s | +0.450996s |
| 16/8/4＋布局（r091） | 0.939299 | 0.255064s | +0.089701s |
| 新4路＋布局（r091，存在实现混杂） | 0.880543 | 0.258906s | +0.419585s |
| 16/8/4＋原布局（r091，不是纯尾部因果对照） | 1.423417 | 0.000006s | -2.404919s |

估算使用r087无插桩归一化重算成本5.679774793s × (1−加权组件耗时比)，减本轮实测准备/逐出/释放。尚未包含实际整模型上下文、初始化及调度交互，也没有候选整请求计时；**表内数字绝不能写成真实整请求已节省。** 即使不扣准备，r092当前组件结果映射的条件收益也仅约0.698s。预先门槛为净潜力>1s，两个主要候选均未过，按约定停止。

## 成本与边界

- 两轮各72个实际模块，BF16原数据共456MiB。每个带布局实现都构建72次、累计分配467.625MiB，单套活跃峰值63.656MiB，命中327次（300次原调用重用＋27次组件计时调用），逐出62次，没有同模块重建或超预算旁路。增加缓存不会减少这条请求的复制次数；不盲目调容量。
- r091同时维护两套带布局的私有缓存，r092只有一套；不能把诊断内存当候选整模型内存。全轮cgroup峰值分别6,453,899,264 / 6,455,914,496字节，包含初始化/预热/隔离检查；不能用该高水位的前后差当派生缓存增量，更不能与RSS相加。
- 逻辑缓存清空后live_bytes/entry均0；CPU分配器与NPU共享内存仍可能影响RSS/驻留量，不宣称释放后RSS立即归零。
- r091原路径带私有校验/配对68.126755s，r092为53.942277s，**都不是候选推理延迟**。
- 新库反汇编K环：compute4 `0x808–0x878`（29条）、compute8 `0x8e0–0x9b0`（53条）、compute16 `0xa44–0xc04`（113条），均无SP访存；16路ABI保存/恢复在K循环外。本轮未新增PMU，不能将剩余差异定量归因于调度、缓存或寄存器压力。
- 返回的原路径响应与合格r072相同，CPU非confidence最大差9.7483e-6、choice/noul/usage/model一致；候选写私有比较缓冲，不等于完成候选整模型/全样本准确率验收。
- 两轮控制器exit0、OOMFalse、资源关闭0、原同一服务恢复healthy；父模型True/Torch3/highest/autocastFalse。原门槛未改。未部署，完整1,000条和定时任务仍暂停。

## 源码、运行与审核

`selected_layout_tail.c`和`build_selected_layout_tail.sh`是两轮共同的实际编译源码；`r091/`、`r092/`分别保存实际测试的Python实现、检查、preflight和probe覆盖集合。每轮需平铺在独立冻结目录，配合父目录既有依赖和[已拒绝ILP](../rejected-selected-dot-ilp/README.md)中的4路调试参照；不能在当前服务目录直接加载。活动probe已按SHA恢复，原计算库保持。

`summary-r091.json`/`summary-r092.json`含可公开计数、逐几何组件原始配对、权重选点量及审核结果，可以复算加权比值；不含问题、标签、输出概率、权重或激活。完整源pin、原始结果、控制混杂审查、恢复证据在忽略目录`output/decision-index-audit/iterations/r091`、`r092`。板端两轮根目录保持冻结。

实际执行：

```sh
# 独立r091目录内，只构建新库；原库未重编译。
bash /opt/decision3-official-opt-20261019-r091/build_selected_layout_tail.sh
caffeinate -i python3 official-long-input-experiment/execute.py --iteration r091 --case selected-layout-preflight --variant full-batch-sparse-request-scope --budget 120
# 等DONE/原服务healthy，再执行r092；它继承完全相同的native二进制。
caffeinate -i python3 official-long-input-experiment/execute.py --iteration r092 --case selected-layout-original4-preflight --variant full-batch-sparse-request-scope --budget 120
```

只读核验与报告重建：

```sh
python3 official-long-input-experiment/review_selected_layout_preflight.py output/decision-index-audit/iterations/r091/20261020T133445-selected-layout-preflight-full-batch-sparse-request-scope-plain
python3 official-long-input-experiment/review_selected_layout_preflight.py output/decision-index-audit/iterations/r092/20261020T134215-selected-layout-original4-preflight-full-batch-sparse-request-scope-plain
python3 official-long-input-experiment/build_selected_layout_trial_report.py
python3 -m compileall -q official-long-input-experiment/rejected-selected-layout-tail
```

本地仅做语法/静态核验；ARM库与真实Owner检查均在板端隔离子进程和上述预检中执行。原模型和数据pin沿用[官方评测说明](../../official-evaluation/README.md)，不是新抽样或新参照。
