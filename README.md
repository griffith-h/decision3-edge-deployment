# d3-edge on RK3576

RK3576 上部署 `vllm-sr/d3-edge` 的研究记录、RKNN 矩阵加速适配和独立安装文档。

## 给接收方的安装文档

下载并在浏览器打开 [从零安装与使用手册](d3-edge-rk3576-install-guide.html)。**只转交这一个 HTML 文件即可**：它内置全部适配源码、固定测试样本及源码初始化脚本，包含系统准备、依赖与模型下载、镜像构建、启动、调用、复现测试和排障步骤。

文档面向与测试机相同的 RK3576、16 GB 内存、ARM64 Armbian / Debian 13。它不依赖原测试服务器的 IP、模型缓存或现成镜像。接收方需自行安装并完成验收。

已有适配核心代码通过了原板端测试；独立从零安装路径尚未完整重跑，后续测试构建按用户要求停止。文档通过了命令语法、内嵌源码还原及桌面 / 手机页面检查。

## 验证结论

采用 **CPU + RKNN MatMul**：将 78 个文本线性层的矩阵乘交给 NPU，保留原始非因果注意力、GatedDeltaNet 因果扫描、视觉塔与 readout。保留官方原始 12 层结构及权重，没有额外裁剪。

| 2026-10-13 测试 | 结果 |
| --- | --- |
| 文本模型内成对基准 | 约 2.19–2.74 倍提速 |
| 小图片模型内基准 | 约 1.43 倍提速 |
| 单问题 HTTP | 6.686 → 2.822 秒，约 2.37 倍 |
| 优化 CPU → RKNN 输出差异 | 5 个用例、12 个问题选项一致；概率 / score 最大绝对差 0.00283136，容差 0.01 |

不同计时口径分别比较。概率差不等于任务准确率下降；结果只覆盖已测样本。超过 512 矩阵行回退 CPU，长文本吞吐和视频性能未验证。

官方 RKLLM 1.3.1 的本次适配能运行 NPU，但没有正确复现非因果注意力，因此不能作为已验证部署方案。`rkllm-d3-support/` 中的导出与策略组件没有接入官方 Runtime，不构成原生修复。

完整方法、原始数据与限制见 [HTML 测试报告](d3-edge-rk3576-report.html)。GitHub 文件页不执行 HTML；下载到本地后用浏览器打开。

## 目录

- `d3-edge-rk3576-install-guide.html`：可独立转交的安装手册，包含源码附录。
- `d3-edge-rk3576-report.html`：历史测试结果及运行状态核验快照。
- `build_install_guide.py` / `build_html_report.py`：两份 HTML 的生成器。
- `rknn-matmul-experiment/`：已测 RKNN 适配、微基准、边界及完整模型测试。
- `rkllm-experiment/` / `rkllm-d3-support/`：RKLLM 转换、诊断与未集成组件。
- `d3-edge-rknn-handoff/`：编写手册期间的独立安装源码草案；`manage.py` 仅通过有限单元检查，不代表安装流程验证完成。请以独立 HTML 手册为交付入口。
- 根目录与各实验的 JSON / 日志：保留的测试证据；IP、路径和服务状态均为当时环境快照。

## 固定版本与下载内容

- 模型：[`vllm-sr/d3-edge`](https://huggingface.co/vllm-sr/d3-edge/tree/20c745653400d191ecacd99c4eafc0c5900dc15e)，revision `20c745653400d191ecacd99c4eafc0c5900dc15e`。
- RKNN：[`airockchip/rknn-toolkit2`](https://github.com/airockchip/rknn-toolkit2/tree/2c723177a94fac65cbc036d722907ddbc91980cf)，Runtime 2.5.0；原板端驱动 0.9.8。
- Python 环境：Torch 2.12.0+cpu、Torchvision 0.27.0+cpu、Transformers 5.17.0；其余固定版本见 `d3-edge-rknn-handoff/requirements.lock`。

模型权重、RKLLM 转换产物、下载的 SDK 二进制 / 手册、浏览器缓存和运行数据不提交到 Git。安装文档中的工具从固定官方来源下载所需文件，并执行相应哈希检查。模型、基础镜像和 SDK 遵循各自许可；SDK 下载工具保留官方 LICENSE。

旧实验脚本含原测试机的固定目录，需先准备其依赖和模型才能运行；克隆本仓库不代表这些历史目录已经存在。

## 本地文档检查

```bash
python3 build_install_guide.py
python3 build_html_report.py
python3 -m unittest discover -s d3-edge-rknn-handoff -p 'test_manage.py' -v
bash -n rknn-matmul-experiment/reproduce-rknn.sh
```

以上不执行板端部署或完整性能测试。真实 NPU 验证和基准按安装手册在接收方自己的机器执行。
