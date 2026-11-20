# Decision Index 官方来源分层抽样评测

本目录归档 2026 年 10 月的 d3-edge 评测。整批 1,000 条已暂停，原 graph-half 服务在该轮结束时恢复，候选没有部署。以下计划与服务状态均对应当时的开发记录。[十月长输入结果](../official-long-input-experiment/CURRENT.md) · [项目最近状态](../WORK_STATUS.md)

十月的开发流程先冻结慢请求，完成定向优化与保护回归，再扩展覆盖。阶段目标为已知长输入 ≤60 秒、两问题 ≤30 秒、32 问题 ≤120 秒。r011 在三类已知慢请求上各有单次达标结果。最终固定同一版本跑完 1,000 条的验收仍待完成。[循环执行计划](PLAN.md) · [已知慢请求](LONG_INPUT_OPTIMIZATION.md)

**当前范围已按用户指令改为只评测最后的 native BF16 优化版本。** 取消最初版本和其余历史阶段的正式对比；同一份 1,000 条冻结样本不变。CPU 部分记录保存在 `output/decision-index-audit/runs-historical-stopped-20261019/`，不与最终版本结果合并。最终版本测完、评审并完成接口验收后部署；自动定时检查保持停用。

本目录是新一轮评测工具，旧文章中的速度和 34 例回归结果不自动变成此次结果。执行状态和原始数据保存在忽略的 `output/decision-index-audit/`；汇总完成之前不宣称各阶段准确率或延迟已测完。

## 数据与版本

- [Decision Index 复现工具](https://github.com/apolinario/decision-index/tree/f5a6e4fd85fed05e77878dfc4966d0568bfb45ec)，固定提交 `f5a6e4fd85fed05e77878dfc4966d0568bfb45ec`，edition 0.3。它是模型发布方采用的第三方评测，公开部分与榜单私有部分分开。
- 模型固定 `vllm-sr/d3-edge@20c745653400d191ecacd99c4eafc0c5900dc15e`。所有阶段保留原 12 层与权重；执行方式和适用形状按历史实现。
- 从上游固定源重建，原始、0.2 裁集、新增基准与 0.3 GSM8K 分别核验 SHA256。HLE 必须先用有访问权限的 Hugging Face 账号登录。
- CFColor 原站当前连接超时；可使用原站的公开存档 `https://web.archive.org/web/20161207205250id_/http://www.dgp.toronto.edu/~donovan/cfcolor/cfcolor.zip`，但必须匹配上游固定 SHA256 `47c07095642cfab3c2eeab366a5d152b07783cbb5af390cbfda7d7c13db4b54c` 才接受。ESCI 的 LFS 文件由上游代码从固定媒体链接下载并验证，缺少 Git LFS 时只在本地下载缓存中修复检出。
- 当前 SGD 构造器已修复历史泄漏，冻结文件哈希仍对应旧格式；RouterBench 的均值序列化需要 Python 3.12 的浮点求和行为。`restore_frozen_builders.py` 用上游历史提交 `2d573f852627b7e1e3180d4b6549c90b75e81b29` 的原 SGD 构造器和当前原 RouterBench 构造器恢复三个规范化文件，并要求哈希匹配。SGD、RouterBench 按官方 0.3 规则不计入主指数；本工具的额外单标签准确率也只统计官方计分面板。
- 不将原始数据、带题目的结果、账号令牌或下载缓存提交 Git。

## 从获取到冻结

仓库根目录，独立 Python 3.12 环境：

```sh
mkdir -p output/decision-index-audit
git clone https://github.com/apolinario/decision-index.git output/decision-index-audit/kit
git -C output/decision-index-audit/kit checkout f5a6e4fd85fed05e77878dfc4966d0568bfb45ec
uv venv --python 3.12 output/decision-index-audit/venv
uv pip install --python output/decision-index-audit/venv/bin/python -r official-evaluation/requirements.lock
uv pip install --python output/decision-index-audit/venv/bin/python -e 'output/decision-index-audit/kit[rebuild]'
output/decision-index-audit/venv/bin/hf auth login
output/decision-index-audit/venv/bin/python official-evaluation/prepare_sources.py --include-hle
```

获取完成后使用上游的标准化函数。完整重建命令如下，会再次检查下载缓存并运行标准化：

```sh
output/decision-index-audit/venv/bin/python -m decision_index suite rebuild --edition 0.3 --work output/decision-index-audit/work
output/decision-index-audit/venv/bin/python official-evaluation/restore_frozen_builders.py
output/decision-index-audit/venv/bin/python official-evaluation/finalize_corpus.py
output/decision-index-audit/venv/bin/python official-evaluation/freeze_sample.py
```

第一次执行中，部分标准化函数在等待授权期间已先运行；这不改变最终上游冻结与哈希校验。已有规范化文件可用 `finalize_corpus.py` 收尾，避免重复重建昂贵的检索索引。

`freeze_sample.py` 默认目标 1,000 条、种子 20260919，直接调用上游分层采样器并保留完整题组；实际条数允许因最后一个题组略超目标。冻结前不看任何模型输出。保存请求 ID、组、载荷哈希和文件哈希。拒绝覆盖已有样本。

## 历史阶段与执行

`stages.py` 列出 17 个对照：原始 BF16 CPU、CPU 适配、初版 NPU、三线程 CPU 优化、融合并行、扫描、安全 FP16、投影合并、布局融合、BLAS2 扫描、graph-half、多长度 MLP、长投影、长序列算子、长尾调度、精度/readout、最新 native BF16 MLP。未通过正确性验证的 RKLLM 路线不冒充有效优化版本。

目标 RK 主机须具备最终版本依赖的历史工作区、固定镜像、模型和编译产物，路径由 `stages.py` 明示。所有历史挂载只读，当前评测代码单独放在 `/opt/decision3-official-evaluation-final-20261019`；原跨阶段评测目录保留归档，不覆盖。

```sh
rsync -az --exclude=__pycache__ official-evaluation/ root@192.168.1.164:/opt/decision3-official-evaluation-final-20261019/
output/decision-index-audit/venv/bin/python official-evaluation/run_all.py \
  --sample output/decision-index-audit/sample-1000.jsonl.gz --stop-service --stages native-bf16
```

该命令先保存并核对原服务容器身份，暂停原服务，只评测 `native-bf16`；退出时重启原容器并等待健康。它不自动将未评审的实验版本上线。用户已授权临时替换服务；最终版本测完之后还需完成接口验收、部署与健康核验，不能以恢复旧服务代替此任务。定时自动检查已停用。

评测器对样本内容、模型版本和本目录代码做身份检查，拒绝在协议变化后混合断点结果。每阶段一个模型主线程和独立容器，CPU4–7、8GiB；线程数记录在阶段配置中。上游运行器接收原 `state` 与 `questions`，标签不传给模型。通信采用持久 SSH RPC，不把网络往返当作纯推理时间。

只在首次运行前同步目录；正式运行期间不要再次 rsync 到同一挂载目录。板端身份包含该目录中的源码、历史依赖与嵌套调度计划哈希。代码修订须停止控制器、等待恢复旧服务，将旧运行目录单独归档后再开始新协议，不能拼接两种协议的结果。首次预运行因发现缺少统一预热而中止，记录保存在 `runs-prewarm-omission-audit`，不纳入正式对比。

当前冻结样本为 1,000 请求 / 831 题组 / 43 基准 / 8,508 问题；内容 SHA256 为 `57d0e3fdd655741beacc5cf492131aa061106d63a613a3f9530c1356bc57c35d`。每请求上限 600 秒；超时按错误记录并重建相同身份的工作进程后继续，不重抽样、不裁剪长输入。

## 怎么读结果

- `results.jsonl`：上游格式的逐次输出、错误、通信墙钟，另有 `raw_output.board_method_ms` 板端模型方法耗时及计数。
- `board-identity.json`：实际源码、配置、库和图哈希，以及软件、线程、模型身份。
- `sample-summary.json`：样本是否全部完成、成功/不支持/错误数、板端与通信 P50/P95、按输入长度分组的耗时、原生任务指标。
- `labelled_field_accuracy`：仅明确单标签 choice 与布尔 noul 字段；失败和不支持也保留在分母。排名、群体偏好和连续真值另按各自原生指标评估。
- `sampled_public_index`：只有抽样覆盖所需基准时才按上游公式计算，明确是样本统计，不是完整公开集或官方榜单成绩；缺项时不重分配权重冒充完整指数。

同一样本对所有阶段使用同样顺序，不裁剪输入或更换选项，不删除慢样本。模型加载与初始两次统一预热排除；样本里首次形状的准备成本保留，因此称为模型已加载的混合请求评测，不能说每种形状都已充分暖机。失败/不支持率必须与成功请求速度一起读。

本地刷新汇总页面：

```sh
output/decision-index-audit/venv/bin/python official-evaluation/build_comparison.py
```

生成忽略目录下的 `comparison.html` 与 `comparison.json`，只包含汇总数据。尚未测完的阶段不展示准确率结论；完整版本另提供与 CPU 适配版在共同成功请求集合上的配对速度。

## 全部完成后的部署

以下步骤尚待正式结果完成后执行，不能把脚本已写好当作部署已验证。

1. 确认最终 `native-bf16` 完成同一批 1,000 请求，检查准确率、原生任务指标、失败率与速度，再生成汇总 JSON。`final_stage_evaluation_complete` 表示本轮最终版本测完；`all_stages_complete` 仍为 false，不伪称取消的 17 阶段已完成。失败请求完成了执行尝试，并不等于模型成功答题。
2. 将最终汇总、native 阶段 `board-identity.json` 与部署工具复制到板端**新工具目录**，不要覆盖冻结评测目录。运行 `prepare_release.py --verdict ... --identity ...`，验证实际评测源码与复制文件的哈希，生成独立只读发布包。
3. 运行 `launch_release.py stage --root ...`，在 8901 验证；运行 `verify_final_http.py --root ... --port 8901 --output ...` 检查 34 例输出、原 0.01 数值门槛、confidence 独立差异、422、模型列表和不同请求并发隔离。不能将这些数值回归称为新增业务准确率。
4. 记录结果评审 `acceptance.json`：`accepted: true`、具体 `reason`、汇总文件 `verdict_sha256` 与 `sample_rows_sha256`。这是完成实际评审后的证据记录，无须重复请求用户授权。
5. `launch_release.py activate --root ... --proof ... --acceptance ...` 才切换 8900；启动失败会恢复保留容器及其重启策略。随后在 8900 再运行 HTTP 检查，留存健康状态、镜像、只读挂载、发布哈希与计数。`launch_release.py rollback --root ...` 可回到原服务；原容器和包保留。

发布限定为相同权重与原 12 层、同一候选 Owner，Torch3 / scan4、CPU4–7、8GiB。不得在部署时另加截断、改模型或把样本失败隐藏在成功请求速度中。如果结果暴露精度或稳定性问题，先处理问题并重测，不能仅因跑完就上线。

## 定向检查

```sh
output/decision-index-audit/venv/bin/python -m unittest discover -s official-evaluation -p 'test_*.py' -v
python3 -m py_compile official-evaluation/*.py
git diff --check
```

当前单元检查覆盖 RPC 读缓冲/原生日志、标签不进入推理载荷、统一预热、重复样本 ID 拒绝、失败样本分母、按抽中样本调用原生评分器，以及发布文件变动拒绝、容器归属与失败恢复。板端正式评测、发布包和 HTTP 验证仍需真实执行。
