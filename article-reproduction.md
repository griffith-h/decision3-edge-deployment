# 文章的测试来源与复现说明

本文对应 [图解文章](d3-edge-inference-engine-optimization.html) 与 [Markdown 原稿](d3-edge-inference-engine-optimization.md)，数据截止 2026-10-18。实验源码与原始结果可按仓库提交 `d724ced162d00dee1625ecb825fc300eab793793` 核对；本文是后补的阅读入口，不改变实验记录。

这个日期以后，d3-edge 又冻结了 1,000 条官方来源的请求，d3.1-nano 另用 TypeSafe 和 vLLM 的 13 条公开请求做了板测。两轮的模型、样本和参照分别记录。[d3-edge 官方来源评测](official-evaluation/README.md) · [d3.1-nano 八核结果](d3-engine/EIGHT-CORE.md)

## 1. 是否使用官方测试集

**文章主结果使用仓库自建的固定性能请求与回归集，没有运行官方完整准确率基准。**

[固定版本的官方模型卡](https://huggingface.co/vllm-sr/d3-edge/blob/20c745653400d191ecacd99c4eafc0c5900dc15e/README.md#evaluation)列出了 Jev Decision Index、图像与视频评测。这些是模型发布方的评测，不是本仓库优化引擎后的复测结果。

仓库早期还保留了 [golden-reference.json](golden-reference.json) 与 [golden_test.py](golden_test.py) 做少量接口输出对照。这个文件名不表示官方完整测试集，也不是本次 34 例回归的统一参照。本次 34 例参照由独立运行的原 `runner.Owner/current` 路径生成，属于冻结的历史引擎输出，不是人工标签，也不能直接称作纯官方 BF16 实现的输出。

## 2. 样例、计时与回归分别是什么

| 项目 | 输入与来源 | 本文结果的含义 |
| --- | --- | --- |
| 短输入性能 | [request.json](cpu-tail-deployment/request.json)，自建中文分类请求 | 同一请求重复 100 次，暖态 LAN HTTP 中位数 325.27ms；不是 100 个不同任务 |
| 长输入性能 | [references.json](fusion-pipeline-experiment/logs/references.json) 中 `cases.words-512.request` | 同一构造请求，在三轮独立容器中各做 30 对基线 / 候选交错计时；候选共 90 次，模型方法中位数 1579.62ms |
| 长输入候选回归 | 同一 `references.json` 的全部 34 条请求；[构造代码](fusion-pipeline-experiment/make_references.py) | 与冻结 original current 的输出差异，检查选择、概率、score、noul；没有业务真值标签 |
| 短输入部署验收 | [http-cases.json](cpu-tail-deployment/http-cases.json)，19 场景 / 26 问题 | 接口输出、形状命中 / 回退与隔离检查；与后续实验的 34 例分开 |

短输入原文：

```json
{
  "model": "vllm-sr/d3-edge",
  "state": "请写一个 Python 函数，合并两个有序列表。",
  "questions": {
    "kind": {
      "type": "choice",
      "instructions": "这个请求属于哪类任务？",
      "criteria": {"code": "编程任务", "math": "数学问题", "chat": "日常聊天"}
    }
  }
}
```

长输入保留相同的 `questions`，把 `state` 替换为 `' '.join(['alpha'] * 512)`。`words-512` 指 512 个重复词；经过模型对输入与问题的格式化，实际 `usage.input_tokens` 为 **615**，不是正文有 615 个词。这是固定长度性能样例，不代表真实长文本分布。仓库另有自然长文本等回归用例，但本文 1.58s 不是这些不同输入的平均耗时。

34 例组成：历史 9 例（含长短输入、多问题、图片）+ 1 例 `wide2-natural` + 12 例替换词的 615 token 输入 + 12 例额外短输入。其中 13 条为 615 token；这些重复词变体不能视作覆盖 13 种真实业务。

数值门槛为选择项不变，概率 / score / noul 最大绝对差 ≤ 0.01；`confidence` 单列，最新 34 例实测最大差约 0.01415。**门槛通过不等于准确率不下降。** 当前证据只支持“已测回归输出符合原数值门槛”。

## 3. 不运行模型，先重算文章中的统计

从仓库根目录运行以下命令，只读已有 JSON，不连接设备、不加载模型、不改写结果：

```bash
python3 - <<'PY'
import json, math, statistics
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text())

def report(name, values):
    values = sorted(values)
    print(name, 'n=', len(values),
          'P50_ms=', round(statistics.median(values) * 1000, 2),
          'P95_ms=', round(values[math.ceil(len(values) * .95) - 1] * 1000, 2))

http = read('cpu-tail-deployment/production-http.json')
report('short HTTP', [r['seconds'] for r in http['records']])
formal = [read(f'bf16-pipeline-experiment/logs/formal-{i}.json') for i in (1, 2, 3)]
assert all(r['completed'] for r in formal)
for mode in ('baseline', 'candidate'):
    report('long ' + mode, [x['seconds'] for r in formal for x in r['records'] if x['mode'] == mode])
refs = read('fusion-pipeline-experiment/logs/references.json')['cases']
reg = read('bf16-pipeline-experiment/logs/regression-native.json')
assert reg['completed'] and len(refs) == len(reg['cases']) == 34
assert set(refs) == set(reg['cases'])
assert all(x['vs_current']['passed'] for x in reg['cases'].values())
print('regression: 34 retained cases passed recorded numerical checks; no labelled accuracy result')
PY
```

预期统计（单位 ms）：

| 计时 | 次数 | P50 | P95，nearest-rank |
| --- | ---: | ---: | ---: |
| 短输入 LAN HTTP | 100 | 325.27 | 328.92 |
| 长输入上一合格候选，模型方法 | 90 | 1743.55 | 1770.60 |
| 长输入最新候选，模型方法 | 90 | 1579.62 | 1642.71 |

上面的回归检查读取已记录的判断；如需逐字段从原始输出重算门槛，阅读并运行 [validate_records.py](bf16-pipeline-experiment/validate_records.py)。该脚本不加载模型，但会重写 `logs/summary.json`，运行前保留原文件。重算统计和输出差异只验证已有记录，不等于重新执行推理。

早期约 6.69s 来自 [api-verification.json](rknn-matmul-experiment/logs/api-verification.json) 的 `cpu_preservation.seconds`，仅单次 HTTP 抽查；初版 2.82s 为 3 次中位数。不同样本量与计时边界不能用来证明严格的总体提速倍数。

## 4. 在同类设备上重新跑短输入 HTTP

前提是已按 [基础安装手册](d3-edge-rk3576-install-guide.html)、[graph-half 构建说明](cpu-tail-experiment/REPRODUCTION.txt) 和 [部署说明](cpu-tail-deployment/README.md) 准备好固定版本、模型与图，并启动待测 graph-half 服务。模型 revision 为 `20c745653400d191ecacd99c4eafc0c5900dc15e`。

`verify_http.py` 当前把测试地址写在 `base` 变量中。先将它的主机地址改为自己的设备 IP，保留测试请求、期望输出和门槛，再在开发机仓库根目录执行：

```bash
python3 cpu-tail-deployment/verify_http.py --port 8900 --count 100 --output reproduction-http.json
```

新结果写入 `cpu-tail-deployment/reproduction-http.json`。脚本先做 19 场景验收和预热，再计时同一条 `single` 请求 100 次，还检查部署身份、线程、图命中及输出；并非任何兼容 API 服务都能直接使用该脚本。不要把失败的断言删除后仍称作同一协议。一次 `curl` 只能验证接口调用，不能替代 100 次暖态统计。

## 5. 重新跑长输入实验与 34 例回归

先按 [第六阶段复现文档](bf16-pipeline-experiment/REPRODUCTION.md) 准备固定工作区、镜像、历史依赖、模型、ARM 库与图，并完成构建和定向验证。该文档还链接了相关历史阶段。`run.sh` 使用固定的 `/opt/...` 目录挂载，不能把命令直接粘到空白机器上运行。

在自己的板端实验目录，先备份原始 `logs/` 到目录之外；下面的脚本会覆盖同名日志。顺序执行，每次等待前一个容器退出：

```bash
bash run.sh regression_native.py
bash run.sh formal_native.py --round 1 --pairs 30
bash run.sh formal_native.py --round 2 --pairs 30
bash run.sh formal_native.py --round 3 --pairs 30
```

正式长输入每模式先预热 3 次，每轮 30 对按先后顺序交替执行，三轮各启独立容器；所有正式样本保留，P95 为排序后第 `ceil(0.95 × n)` 项。不计模型加载，不含 HTTP，计时期间不并行做编译、哈希扫描或性能剖析；每次都实际推理，不读取参考答案返回。冷启动、并发和其他内容的速度另测。

## 6. 当前可复现程度与限制

仓库已保存请求原文、构造代码、冻结输出、逐次延迟和执行脚本，读者可审计并重算文章数字。**完整的新机器复现仍需准备历史工作区并下载或重建未入 Git 的模型、SDK、ARM 动态库、RKNN 图和捕获文件；现有材料不是从零一键复现包。** 各阶段文档中的固定镜像与绝对路径也是环境条件，应逐项核对。

单独转发文章 HTML 时，其样式和图表可以离线查看，但链接的 JSON、代码及本说明不在 HTML 内。论坛发布时应同时附上[完整仓库](https://github.com/griffith-h/decision3-edge-deployment)，并确保文章与本文也已推送。

这些测试支持已列场景的性能和输出回归结论。若要声称“官方基准准确率不降”，还需要对官方标注测试集分别运行参照引擎和优化引擎，按同一评测协议比较；本次没有完成这项验证。
