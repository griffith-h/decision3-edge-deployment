# 原4路＋缓存友好布局：完整请求配对

2026-10-20，r093。用户要求补齐完整请求实测，不再以r092组件条件估算决定是否运行配对。r091/r092的历史组件结果保留，不将其改写为完整请求收益。

## 执行前固定的验收条件

- 固定同一Owner、模型checkpoint、12层、B2/T1193请求与原批次/padding。原4路`EnabledSequentialRepair`调用同一个未重编译的`libsparse_cpu.so`；新旧只切换权重行布局。+64B行间隔与64MiB请求内LRU预算固定，不扫参数，不启用16/8路。
- 保留原615/115保护预热，再各做一次原/新完整B2形状预热；新路径每次从空缓存开始，不跨请求留副本。随后依次原→新、新→原、原→新，共3组，不丢弃慢值。
- 请求计时包围完整`model.system_one`，包含布局复制、逐出、缓存命中管理及结束释放。安装耗时另测，保守地从每组收益中全额扣一次；不将初始化成本摊薄。记录准备/释放分项、精确缓存峰值/累计分配、cgroup与RSS，不把累计高水位解释为单个模式内存增量。
- 每个新/旧完整响应须与原形状预热及已合格r072完全相同，原CPU非confidence≤0.01、choice/noul/usage/model保持，后端True/Torch3/highest/autocastFalse，选点/修复计数一致、无密度拒绝/读取错误/CPU旁路。原选点、K求和顺序、2.5e-7经验带及保护1e-6不变。
- 三组净节省（原耗时−新耗时−一次性安装）的中位数须**严格大于1秒**，才跑固定同上下文完整34例与原8/8/2完整18题。未过就停止，不以最快一次代替三组结论，不部署、不启动1,000条或定时任务。
- 原生检查在隔离子进程执行；完整请求中不叠加组件重放、SDK插桩或逐点位模式检查。沿用r092所有真实选点逐位核验与冻结内核，保留完整响应/调用计数验证。结束必须确认资源关闭、缓存/方法恢复、原同一服务恢复healthy。

## 实现与复现

`selected_layout_tail.py`与其五项测试逐字继承r092；`OriginalLayout`只调用原4路。`original_layout_scope.py`为本轮请求边界和开关，新增一项真实ARM测试覆盖原内核、冷缓存、成功/异常清理及方法恢复。`original_layout_pairs.py`计时与记录；`probe.py`是在活动r087 probe上增加这一个实验分支，活动probe不修改。

在独立r093目录继承r092全部顶层文件和inputs，再覆盖本目录六个Python文件；不编译任何库。控制器创建只读源码快照，停旧服务、串行执行、关闭实验并恢复原服务。私有输出仅留忽略目录。

```sh
caffeinate -i python3 official-long-input-experiment/execute.py --iteration r093 --case original-layout-pairs --variant full-batch-sparse-request-scope --budget 120
```

每个完整B2请求上限120秒，预热单列；本结果仅针对固定两问题开发请求，不能代表完整官方榜单或所有请求延迟。

## r093实际结果

3组原/新完整耗时分别31.325268/30.842188、30.687463/30.750240、30.624519/31.817208秒。扣一次性导入/安装0.013851秒后，净收益+0.469229/−0.076627/−1.206539秒，中位−0.076627秒；未过>1秒门槛，按约定停止，没有执行此候选34例、18题、全1,000条或部署。保留第三组慢值，不以组件估算或最快一次替代结论。3组只说明本实现未证明所需收益，不能断言布局方向永远无效。

81项ARM隔离检查通过，2次完整形状预热＋6次计时请求的完整响应与合格r072完全相同，原CPU非confidence最大差9.74830259e-6，离散判定/usage/model一致。每次372次修复、8,063,030选点，计数一致、无错误/旁路/密度拒绝，后端保持True/Torch3/highest/autocastFalse；数值门槛源SHA保持841c6b517241d3d9af051bcd18a2f51a83c2d30a4aef4b05eaec1e85f1195683。未改选择器或原4路ELF。

每次新请求72次复制构建、300次命中、62次逐出；活跃payload峰值63.65625MiB，累计派生分配467.625MiB，请求结束条目/live_bytes归0。准备/复制/逐出三次中位0.260408秒、释放中位0.003144秒，已包含在完整耗时，不重复扣除。全轮cgroup峰值6,766,145,536字节，包含初始化/预热/隔离检查，不是独占增量；原始RSS/cgroup快照见公开汇总。无OOM。

控制器run20261020T152924-original-layout-pairs-full-batch-sparse-request-scope-plain，DONE completedTrue/service_healthyTrue/exit0。缓存、方法与句柄清理检查通过；原同一容器07577e5f9827恢复healthy。活动probe及数值核心从未修改，候选只在独立r093冻结目录执行。

[完整图表](../SELECTED_LAYOUT_REQUEST_PAIRS.html) · [Markdown](../SELECTED_LAYOUT_REQUEST_PAIRS.md) · [原始可公开汇总](summary-r093.json)。复核命令：

```sh
python3 official-long-input-experiment/review_original_layout_pairs.py output/decision-index-audit/iterations/r093/20261020T152924-original-layout-pairs-full-batch-sparse-request-scope-plain
python3 official-long-input-experiment/build_selected_layout_pairs_report.py
```

公开汇总不含题目、标签或模型输出；完整私有日志、输出、源pin保留在忽略目录。原候选所需旧C/库编译来源见上一轮rejected-selected-layout-tail，本轮不编译任何库；sha256逐字继承检查通过。
