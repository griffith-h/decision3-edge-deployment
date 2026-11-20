# CPU 参照一致性审计（进行中）

另一个参照口径核实：冻结的 `/release/cpu_attention_optimized.py::install_sdpa` 对Qwen文本注意力把query/key/value转FP32再调用原SDPA，随后转回输入dtype。因此r019“原生BF16 SDPA”组件观察不等于仓库实际CPU注意力参照；其FP32分支才接近已安装路径。原始日志保留，r033仿BF16 Flash的失败不应被解释为原CPU路径的性能/精度上限。完整原CPU参照r018/r058不受该组件诊断命名影响。

最新：r060逐项状态记录定位到投影边界测试中的原生BF16 Linear触发Torch警告 `mkldnn_matmul failed, switching to BLAS gemm`，随后主进程MKLDNN变为False。r061将所有检查置于独立子进程：实际69项通过，子进程最终False，父模型进程20个快照全True。它证明环境污染被隔离，尚不证明候选数值通过。r017同样有该警告；其两问题CPU诊断保留为旧False环境结果，重新取得原True后端CPU参照。原r018/r058完整CPU参照与0.01门槛不变。

截至 2026-10-20：完整18题仍未通过原0.01数值门槛，未部署。不能因为某个诊断更接近候选，就直接替换旧CPU参照。

## 已取得的证据

| 记录 | 范围与结果 |
| --- | --- |
| r018 | 正常 `model.system_one` 原8/8/2批次的完整CPU参照；NPU/图计数增量0，仅作正确性参照。 |
| r055 | band2候选完整18题255.34秒，字段正确率11/18与r018相同、choice/noul一致；非confidence最大差0.04268233，仍失败。 |
| r057 | 原第一批B8/T1181，真实CPU逐层重放与候选前11层逐项一致；末层仅1个值不同，位于非末行。 |
| r057读出 | CPU重放和候选末层hidden分别经过同一原末norm、FP32 readout、温度、softmax、respond，首B8答案所有数值差0；也与r055实际候选首B8一致。 |
| 与旧参照比较 | r057 CPU重放首B8与r018实际CPU首B8仍有非confidence差0.04268233；含confidence差0.08333921。重放不等于独立完整CPU调用，须审计。 |

r018/r045/r055/r056的输入文件SHA256均为 `b851b176ad8bda9414d18f90c90ebef287058feb4293d48b5e0bd55427e8a440`；此完整18问题payload的规范化SHA256均为 `63fb36720e86ba189ea4dc7dd9aa0add948cc95bdaed192a834f3546b30eaf6b`。已比较身份记录中的基础运行文件，未发现r018/r045依赖SHA差异。

## 独立复跑的中间结果

r058全部18题已完成：336.84/331.09/80.34秒，仅正确性计时；与r018所有受检数值（含confidence）均差0，choice/noul/usage/model一致，三批NPU/graph增量0。顶层源码/二进制哈希完全相同，关闭0、原服务健康。**旧CPU参照可完整复现，继续保留；此前重放不是等价独立CPU参照。**

## 已完成的独立复核

r058完整复制板端r018的顶层源码、二进制及inputs到新目录，运行当时原脚本的 `cpu-full` 分支，正常调用完整18题的原8/8/2批次，每批原600秒上限。它不安装band2适配器或69项新检查，也不改旧目录；仅独立复核原CPU流程，不恢复17阶段或完整1,000条。

```bash
caffeinate -i output/decision-index-audit/venv/bin/python \
  official-long-input-experiment/execute.py \
  --iteration r058 --case coverage-0000 --variant cpu-full --budget 600
```

## 下一步：实际调用的切换审计

1. 原正常CPU复现已通过。下一轮在同一Owner中先完成候选原首B8，再撤销适配、正常调用CPU `model.system_one` 原首B8；不重放hidden，不改encoder/decoder方法。
2. 若正常CPU稳定而重放不同，定位重放与正常调用的实际流程/上下文差异，修复诊断或引擎；不改数值门槛。
3. 若正常CPU自身不稳定，先复核输入token、运行配置、实际CPU算子与重复运行；保留全部原始结果，查清原因后再确定可复现的参照。
4. 当前不以任何一种假设作为结论，不替换旧基线、不部署。

原始输入、输出和激活只在忽略目录 `output/decision-index-audit/iterations/`；r057的 `readout-consistency-review.json` 明确标记仍须实际CPU复核。

## 正常切换与后端快照

r059不改encoder/decoder方法、不重放hidden，实际CPU `model.system_one` 首B8与候选完全一致；实际token SHA与counts一致，CPU NPU/graph增量0、退出关闭0、原服务健康。但仍与r018旧CPU不同。此前差异并非仅由重放造成。

r059两角色MKLDNN均False。r060独立CPU-full的初始化、615/115预热和切换后均True，其他已记录状态为Torch3/highest/autocastCPU False。正在逐项追踪候选的测试导入/执行及适配安装，找出状态变化的步骤；尚不能把这一相关性当成全部差异的已证实原因，不改门槛、不部署。


## r061 / r062: original enabled backend restored in diagnostic harness

All 69 SDK boundary checks now run serially in a separate child process, with no live model owner in that child. The parent retains MKLDNN=True through adapter installation and actual inference. New same-B2 CPU reference r061 is separate from old False-context r017 (94.4899 seconds, correctness-only). r062 candidate takes 33.2845 seconds and differs by at most 0.007825390976162583, including confidence; original 0.01 and discrete-choice gates pass, 30-second speed target fails. This does not qualify the complete 18-question group; the original full CPU reference and previous full-case failures remain intact.


## r062 / r063: enabled-backend primitive mismatch isolated

r062 original first B8 projection has 8,340 qkv and 3,023 z BF16 mismatches; no full-model conclusion follows from this partial result. r063 samples 512 evenly spaced mismatching qkv positions. With original MKLDNN enabled, Torch BMM4, BMM16 and selected-square Linear each match all 512 original full-projection BF16 values. FP64 sum differs at 295, FP32 elementwise sum at 284, BMM1 at 230. Flattened full CPU Linear matches the original 3D projection at every value. Thus naive sums and the previous public-BLAS-oriented repair are not established equivalents of the enabled-backend CPU reduction. These selected positions are diagnostic intermediates, not a benchmark or production dispatch key. No numerical threshold is changed.

## r076～r078：图片请求使后续历史身份参照处于False后端

r076隔离子进程73项检查后主进程仍True，但34例中的`image`触发固定运行时`_linear_patch_embed_forward`的BF16 F.linear回退，原运行时将MKLDNN全局改为False。该轮34例输出都满足旧冻结门槛，环境隔离却失败。r077在patch结束立即恢复True导致图片整次输出非confidence差0.01308144，拒绝，不能以局部算子相同推断请求相同。

r078只在完整公开请求结束/异常时恢复进入状态。真实图片→B2→615/115连续调用全部通过原数值/选择门槛，请求外一直True；图片对原current差0.001819726，B2对r061原True CPU差9.7483e-6，两个保护形状对历史native差0。

同Owner对`matched615-5/-6/-11`分别用True/False后端、原native logits和candidate logits交叉运行：每种同后端候选与原native所有数值差0；False原native与历史native差0；True原native与历史差3.99572862e-6/1.03794885e-5/1.18696202e-5。源脚本按字典顺序在同一Owner生成旧34例，`image`在这些用例之前；实测复现支持历史输出受图片后端状态影响。**旧冻结文件与旧1e-6/0.01门槛保持；这是参照上下文审计，不把静态检查失败直接改成通过。**

原5组115/615配对输出差0，中位比0.996217/0.989756，关闭0、主后端True、原服务健康。原始12条交叉记录、4条连续调用、20条配对和审核在忽略目录r078；当时尚未完成新上下文完整34例，后续结果如下。

## r080：独立参照冻结后完整34例通过

三组分别使用独立Owner：原CPU、原native、候选。每请求前后固定MKLDNN=True、Torch3、highest、autocastCPU=False；图片内部原fallback保留，请求结束恢复。完整输入及顺序SHA为`5aa01b4b93959a07e51a320648ed6ce0a5805e6fa2f15cef81cab69d05cf297b`。

r079首次检查拦截CPU参照仍有115图调用，未冻结失败输出。r080只在新reference入口明确bank.mode=cpu，34例每例NPU/graph调用增量均0。原CPU/native运行完成、检查输入/源身份/后端后，先冻结参照，再启动候选；未用候选输出生成参照。

候选34/34通过原check_pair：两参照非confidence差≤0.01，选择/noul/usage/model一致，115/615同上下文native全部数值差0，严格1e-6通过。对纯CPU最大非confidence差0.00943617884，含confidence最大差0.01832714332单列。历史CPU/native的0.01和离散门槛也通过；历史False参照3条严格身份失败仍报告，未覆盖或伪称旧静态门槛全过。

新参照SHA为`adeb050676d8baac49083d56dd858128c895a6680fb493199bf58c05247ab254`，旧参照SHA仍为`e29c504f1a9aed914e32bee92af65ca6b6f4aae6fddba30731625c427732fdcf`；原门槛源码SHA仍为`841c6b517241d3d9af051bcd18a2f51a83c2d30a4aef4b05eaec1e85f1195683`。75项ARM检查、资源关闭、服务恢复通过，原始证据在忽略目录r080。这是34例数值回归，不是完整官方样本准确率或部署验收。
