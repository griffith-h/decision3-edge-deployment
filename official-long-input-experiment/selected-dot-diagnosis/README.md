# 选点重算：冻结二进制的访存与执行量诊断

2026-10-20，r089/r090。只诊断r088的8/16路退化，**没有采用新内核或权重布局**。图表和结论见[诊断报告](../SELECTED_DOT_DIAGNOSIS.html)，可编辑正文见[Markdown](../SELECTED_DOT_DIAGNOSIS.md)。

## 固定约束与验收

- 同一D3权重、BF16输入/权重值、真实选点、每输出K0至K-1连续FMA顺序、最终+0及BF16舍入；不重编译原库或r088新库。
- CPU4–7为Cortex-A72，Torch3/OpenMP3，原容器cpuset/内存/环境保持。主机PMU观察进程绑定CPU0–3，只附着到经过Unix peer PID和容器所属关系核验的被测进程线程；不增容器权限、不改sysctl。
- 每组6个计数事件，必须线程集合稳定、有效运行比例≥99%、CPU周期/退休指令非零。实际1040窗口均100%，无复用缩放。主机不装perf，直接调用内核perf_event_open。
- 每份重放先将选中输出填NaN，再执行冻结内核，完整BF16位模式须相同；每个测量窗口结束再核对。指标不达标即失败，不删慢值。
- 本次总诊断预算45分钟；完整1,000条、34/18题候选验收和部署不启动。原0.01/choice/noul/保护1e-6均不变。

## 两轮实验如何分工

| 轮次 | 输入与控制 | 完成结果 |
| --- | --- | --- |
| r089 | 原B2/T1193，原372次重算全部记录选点指纹；捕获每种几何第一次的真实输入，共9种。原库4路与冻结新库4/8/16路，3组PMU事件，各4轮；另对2种形状做+64B行间隔对照。 | 528窗口、80项ARM检查通过。372次/8,063,030选点指纹与r088完全一致；原模型响应与合格r072一致，CPU非confidence最大差9.7483e-6。 |
| r090 | 只重放r089两个已冻结捕获（512×2048→1024、615×1024→3584）。固定2×2：原行步幅/K+32个BF16，与基地址mod64=0/16B。4种冻结内核，core/cache组各8轮均衡轮换。 | 512窗口、全部BF16逐位相同。两种对齐下，+64B行间隔均减少8/16路L1回填与耗时。 |

r089首个padding对照**同时改变了基地址对齐**：原权重偏移16B，padding副本偏移0B。该记录保留，不能单独证明行步幅因果；r090才将两个因素拆开。没有因审核发现混杂而覆盖旧数据。审核脚本对原权重对齐的初始“必须0B”假设已纠正为记录真实0/16B；这不是修改模型数值门槛。

r089的33.269秒包括捕获额外工作，r090不运行完整模型；两者都不是新引擎整请求延迟。PMU逐线程计数先求和，再按重放次数和选点数归一化，不与模型墙钟相加。计时/计数含Python封装、后端守卫与OpenMP，新库4路是封装/编译对照，并非只计纯汇编循环。

## 为什么更偏向L1局部性，而非DRAM带宽或热循环spill

1. 冻结ELF的K循环均无栈读写；16路循环结束后的输出临时数组写栈另外标明。
2. 同对齐/同内核下，K2048的16路行间隔对照：L1回填248.9→69.6/点，退休指令4530.0→4523.8/点，L2回填55.30→55.55/点。新增缓存代价集中在L1/L2之间，没有证据证明DRAM带宽饱和。
3. 仅改基地址0/16B不能消除原步幅的退化；两种对齐下改行间隔都有效。与多条规则步幅权重流引起的缓存组冲突/局部性下降相符，**尚未逐物理地址证明冲突组**。
4. 尾部满宽计算是另一项确定成本。`audit_selected_tail.py`只解析本轮自己生成的Torch存档元数据和int64选点，不加载模型、不执行任意pickle类；核验存档SHA和选点SHA后，统计每行ceil(count/width)并结合实际汇编计数。9种形状16路多算9.3%～25.2%。

CPU端点频率均2.208GHz，DMC仍按原策略自动调频（0.534/1.320/1.968/2.736GHz）。8轮均保留且均衡顺序；不能把时延变化百分比当固定内存频率下缓存因素的唯一占比。未测到硬件停顿分类，不能精确分摊调度、load-use或物理寄存器重命名约束。

## 文件与复现

- `dot_pmu_monitor.py`：板端主机观察器；`selected_dot_diagnosis.py`：r089真实捕获/重放及Unix客户端；`selected_dot_cache_control.py`：r090固定2×2对照。
- `r089/`、`r090/`：各轮实际测试的probe、host launcher及绑定检查覆盖文件；保持冻结版本。依赖父目录活动数值核心与[已拒绝ILP源码](../rejected-selected-dot-ilp/README.md)，需要平铺在独立目录。**不要直接覆盖服务。**
- `pmu-summary.json`、`factorial-summary.json`、`tail-summary.json`、`assembly-summary.json`：经审核的可公开聚合计数与二进制SHA，无题目、标签、权重或激活。
- 完整源码pin、私有捕获、逐窗口/逐线程计数、运行和清理日志在忽略目录`output/decision-index-audit/iterations/r089`、`r090`；板端原快照保留于`/opt/decision3-official-opt-20261019-r089`、`...-r090`。捕获只保存在板端私有0700目录，不复制入仓库。
- 固定原请求payload规范化SHA为`acecf821f479ac8a7af688d9bcfce7affd4bf578c72a3e70f0271cce361603bb`，原样本/数据获取见[官方评测复现](../../official-evaluation/README.md)。先恢复同一权重、运行镜像、r080数值核心及输入，运行r089产生自己的捕获，再将r090 launcher的只读replay路径指向该捕获；不能拿其他激活替代后宣称精确复现。

实际执行（目录与源pin已在运行前固定，不可对原目录覆盖重跑）：

```sh
# 主机PMU观察器先启动；每轮只允许一个控制器。
ssh root@192.168.1.164 'python3 /opt/decision3-official-opt-20261019-r089/dot_pmu_monitor.py --socket /opt/decision3-official-opt-20261019-r089/pmu/monitor.sock --timeout 900'
# 看到PMU_READY后，在另一个终端运行：
caffeinate -i python3 official-long-input-experiment/execute.py --iteration r089 --case selected-dot-diagnosis --variant full-batch-sparse-request-scope --budget 120
# 先等DONE及原服务healthy，再对独立r090根目录启动其观察器和控制器：
caffeinate -i python3 official-long-input-experiment/execute.py --iteration r090 --case selected-dot-diagnosis-control --variant full-batch-sparse-request-scope --budget 120
```

只读审核与报告重建：

```sh
python3 official-long-input-experiment/review_selected_dot_diagnosis.py output/decision-index-audit/iterations/r089/20261020T051005-selected-dot-diagnosis-full-batch-sparse-request-scope-plain
python3 official-long-input-experiment/review_selected_dot_cache_control.py output/decision-index-audit/iterations/r090/20261020T051952-selected-dot-diagnosis-control-full-batch-sparse-request-scope-plain
python3 official-long-input-experiment/build_selected_dot_diagnosis_report.py
PYTHONPATH=official-evaluation python3 -m unittest discover -s official-long-input-experiment/selected-dot-diagnosis/r090 -p test_host_run.py -v
python3 -m unittest discover -s official-long-input-experiment/selected-dot-diagnosis -p 'test_dot_pmu_monitor*.py' -v
```

原冻结库、模型与数值核心SHA未变。两轮控制器均退出0、OOMFalse、清理无错误，原同一容器恢复健康；主机观察器退出并删除socket。活动probe/host及数值核心已按哈希恢复r087，活动host检查保留诊断前已有的SDK隔离测试，诊断覆盖文件仅在此归档。没有新增定时任务。
