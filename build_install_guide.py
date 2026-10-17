#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the single-file handoff document; performs no deployment."""
import ast
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'd3-edge-rknn-handoff'
OUT = ROOT / 'd3-edge-rk3576-install-guide.html'


def esc(text):
    return html.escape(str(text), quote=True)


def code(text):
    return '<pre><code>' + esc(text) + '</code></pre>'


names = ['Dockerfile', 'requirements.lock', 'sdk-manifest.json', 'tools/fetch_sdk.py',
         'tools/download_model.py', 'app/matmul_bridge.cpp', 'app/rknn_linear.py',
         'app/cpu_optimized_server.py', 'app/npu_server.py', 'app/request.json',
         'app/benchmark.py', 'app/micro_test.py', 'app/boundary_test.py',
         'app/request-cases.json', 'app/golden-reference.json']
files = {name: (SOURCE / name).read_text() for name in names}
files['.dockerignore'] = 'data/\nchecks/\n__pycache__/\n*.tar\n*.tar.gz\n'
for name, content in files.items():
    if name.endswith('.py'):
        ast.parse(content)
    if name.endswith('.json'):
        json.loads(content)
bootstrap = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save the source appendix from the handoff document into a new directory."""
import json
from pathlib import Path
files = json.loads(''' + repr(json.dumps(files, ensure_ascii=False)) + ''')
root = Path('d3-edge-rknn')
if root.exists():
    raise SystemExit('d3-edge-rknn already exists; choose an empty parent directory')
root.mkdir()
for name, content in files.items():
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
print('Source files saved to', root.resolve())
'''
ast.parse(bootstrap)
source_details = ''.join('<details><summary>' + esc(name) + '</summary>' + code(content) + '</details>'
                         for name, content in files.items())

body = '''<header><p class="eyebrow">RK3576 / ARM64 LINUX / 2026-10-17</p><h1>在自己的 RK3576 上安装 d3-edge RKNN 加速服务</h1>
<p>适用于与原测试机相同的 RK3576、16 GB 内存、Armbian / Debian 13。本文包含从零操作步骤和全部适配源码，不依赖原测试服务器的 IP、缓存目录或预制镜像。接收方在自己的机器上执行安装和验收。</p>
<nav><a href="#result">方案与效果</a><a href="#prepare">环境准备</a><a href="#source">准备源码</a><a href="#install">构建与下载</a><a href="#start">启动</a><a href="#api">调用</a><a href="#verify">复现测试</a><a href="#trouble">排障</a><a href="#appendix">源码附录</a></nav></header>
<section id="result"><h2>采用什么方案，能快多少</h2><p>使用 RKNN MatMul 将 78 个适合的文本线性层交给 NPU；非因果注意力、RoPE、因果 GatedDeltaNet 扫描、视觉塔和输出 readout 保留原实现。保留官方原始 12 层模型及权重，不额外裁剪，也不需要将整个模型转换为 RKNN 文件。</p>
<table><thead><tr><th>2026-10-13 历史测试</th><th>结果</th></tr></thead><tbody><tr><td>短文本模型内成对基准</td><td>约 2.19–2.74 倍提速</td></tr><tr><td>小图片模型内基准</td><td>约 1.43 倍提速</td></tr><tr><td>单问题 HTTP 请求</td><td>6.686 秒 → 2.822 秒，约 2.37 倍</td></tr><tr><td>优化 CPU → RKNN 输出差异</td><td>5 个用例、12 个问题选项一致；概率 / score 最大绝对差 0.00283136，容差 0.01</td></tr></tbody></table>
<p class="note">概率差 0.00283136 在来自概率时约为 0.283 个百分点，不代表任务准确率下降 0.283%。尚未做完整数据集准确率评测。模型内与 HTTP 计时分别比较；原始 CPU 到优化 CPU 是另一组基线。新机器速度以自行实测为准。</p><p>原生矩阵最多支持 512 行，行数 = batch × 补齐后的序列长度；超出时回退 CPU，不截断输入。长文本吞吐、视频性能和其他 RK 芯片没有本次基准结论。官方 RKLLM 1.3.1 本次路径未正确执行 d3 的非因果注意力，不作为本文安装路线。</p>
<div class="notice"><strong>验证状态：</strong>适配核心源码沿用已经通过原板端测试的版本；本次交付只整理操作文档。本文从零构建路径未完整重跑，独立测试构建已按要求停止。接收方需要完成后文算子、边界和完整模型测试，再判断其机器是否通过。</div></section>
<section id="prepare"><h2>1. 准备自己的 RK3576 系统</h2><p>使用带 RKNPU 内核驱动的板厂 / Armbian ARM64 镜像。原测试组合为 RK3576、驱动 0.9.8、RKNN Runtime 2.5.0，CPU 大核为 4–7。建议有 16 GB 内存，安装前至少预留 15 GB 磁盘空间。Docker 与 Python 包不会替代主机的 NPU 内核驱动。</p>'''
body += code('uname -m  # 应为 aarch64\ntr "\\0" " " </proc/device-tree/compatible  # 应含 rk3576\nls /dev/dri\n# debugfs 已挂载时可查看驱动版本\nsudo cat /sys/kernel/debug/rknpu/version\nfree -h\ndf -h .\n\n# Debian / Armbian 缺少 Docker 时安装\nsudo apt-get update\nsudo apt-get install -y docker.io python3 ca-certificates\nsudo systemctl enable --now docker\nsudo docker info')
body += '''<p>首次构建需要访问 ghcr.io、PyTorch 官方下载站、PyPI、Debian 软件源和 raw.githubusercontent.com；模型下载需要 Hugging Face。网络失败时保留固定版本，修复网络后重试对应步骤。本文不安装或更换主机内核、NPU 驱动。</p></section>
<section id="source"><h2>2. 从本文准备完整源码</h2><p>本文是唯一需要转交的文件。下面按钮只将文末公开展示的源码保存为一个本地 Python 脚本；运行它仅创建源码文件，不安装软件、不联网、不启动服务。可以先阅读脚本，也可以逐个复制附录中的文件。</p><button id="download-init">保存源码初始化脚本 prepare_d3_sources.py</button>
<p>把该脚本复制到自己的 RK 机器，在一个可写目录中执行：</p>'''
body += code('python3 prepare_d3_sources.py\ncd d3-edge-rknn\nexport D3_HOME="$(pwd)"\nmkdir -p "$D3_HOME/data/models" "$D3_HOME/checks"')
body += '''<p>初始化脚本拒绝覆盖已有 <code>d3-edge-rknn/</code>。之后各步均在这个目录运行；重新登录终端后先 cd 回来，再执行 <code>export D3_HOME="$(pwd)"</code>。全部源代码都在本文附录，按钮不从外部网站下载适配代码。</p>
<table><tr><th>文件</th><th>用途</th></tr><tr><td>Dockerfile、requirements.lock</td><td>从固定公开基础镜像构建运行环境，固定已测 Python 依赖版本</td></tr><tr><td>tools/fetch_sdk.py、sdk-manifest.json</td><td>下载官方固定提交的 Runtime / 头文件并检查 SHA256</td></tr><tr><td>tools/download_model.py</td><td>下载固定模型版本并验证完整文件清单</td></tr><tr><td>app/matmul_bridge.cpp、rknn_linear.py</td><td>RKNN 原生矩阵桥接和文本线性层适配</td></tr><tr><td>app/cpu_optimized_server.py、npu_server.py、request.json</td><td>CPU 算子优化、串行模型服务和预热 / 调用请求</td></tr><tr><td>app/benchmark.py、micro_test.py、boundary_test.py 及样本 JSON</td><td>模型、算子和边界复现测试</td></tr></table></section>
<section id="install"><h2>3. 在自己的机器构建镜像、下载模型</h2><p>构建命令不会使用原测试机上的私有镜像。基础镜像固定为公开的 <code>ghcr.io/vllm-project/semantic-router/vllm-sr@sha256:efc356587252c28b8d972d5d27b381df0626ccdc53597f3fccff7ae454041178</code>，之后替换为可在该 ARM CPU 上运行的 Torch 2.12.0+cpu，安装 Torchvision 0.27.0+cpu、Transformers 5.17.0 及锁定的依赖。</p>'''
body += code('sudo docker build --platform linux/arm64 -t d3-edge-rknn:1.0 .\n\n# 下载模型是独立的联网步骤，覆盖镜像的离线环境变量\nsudo docker run --rm \\\n  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \\\n  --mount "type=bind,source=$D3_HOME/data/models,target=/models" \\\n  --mount "type=bind,source=$D3_HOME/tools,target=/setup,readonly" \\\n  d3-edge-rknn:1.0 /setup/download_model.py')
body += '''<p>模型固定为 <code>vllm-sr/d3-edge@20c745653400d191ecacd99c4eafc0c5900dc15e</code>，下载到本机 <code>data/models/d3-edge/</code>。脚本执行 <code>verify_package(..., mode="full")</code>，服务启动时也执行完整文件哈希验证。SDK 固定提交为 <code>2c723177a94fac65cbc036d722907ddbc91980cf</code>；Dockerfile 在容器内部编译 C++ 桥接库，宿主机不需要安装 g++ 或 PyTorch。</p>
<p>本步骤需要等待实际命令成功；镜像构建成功、模型下载成功和 NPU 推理正确是三个不同的验收点。不要跳过下一步真实算子测试。</p></section>
<section id="checks"><h2>4. 先验证 NPU 算子和回退边界</h2><p>在服务启动前运行。每次结果写入新的本机目录：</p>'''
body += code('export CHECK_DIR="$D3_HOME/checks/$(date +%Y%m%d-%H%M%S)"\nmkdir -p "$CHECK_DIR"\nsudo docker run --rm --init --cpuset-cpus 4-7 \\\n  --memory 8g --memory-swap 8g --device /dev/dri \\\n  --mount "type=bind,source=$CHECK_DIR,target=/work/logs" \\\n  d3-edge-rknn:1.0 -c \'import subprocess,sys; [subprocess.run([sys.executable,p],check=True) for p in ("/work/micro_test.py","/work/boundary_test.py")]\'')
body += '''<p>需要命令以 0 退出，且 <code>boundary-test.json</code> 的 <code>passed=true</code>。覆盖 16 个矩阵尺寸、动态切换、127/128/129/256/257/512/513 行边界、超限与非 BF16 的精确 CPU 回退，以及释放资源后恢复 CPU forward。失败时保留日志，先核对 Runtime 与主机驱动，不要仅凭 NPU 有负载就宣布可用。</p></section>
<section id="start"><h2>5. 启动 RKNN 服务</h2><p>下面使用本机 8900 端口，绑定各网卡；接收方自行选择空闲端口及容器名。容器内端口固定为 8900。<code>unless-stopped</code> 会在主机 / Docker 重启后恢复服务，手动 stop 后保持停止。默认不设置 API Key。</p>'''
body += code('sudo docker run -d --name d3-edge-rknn --init \\\n  --restart unless-stopped --cpuset-cpus 4-7 \\\n  --memory 8g --memory-swap 8g --device /dev/dri \\\n  -p 0.0.0.0:8900:8900 \\\n  --mount "type=bind,source=$D3_HOME/data/models,target=/models,readonly" \\\n  --log-opt max-size=10m --log-opt max-file=3 \\\n  --health-cmd \'python -c "import urllib.request; urllib.request.urlopen(\\"http://127.0.0.1:8900/health\\",timeout=5).read()"\' \\\n  --health-interval 30s --health-timeout 10s --health-start-period 180s \\\n  d3-edge-rknn:1.0\n\nsudo docker logs --tail 80 d3-edge-rknn\ncurl --fail --max-time 10 http://127.0.0.1:8900/health')
body += '''<p>加载完成后日志应包含 <code>78 text Linear layers</code>、<code>78 cached NPU handles</code> 和服务 ready；健康接口返回 <code>status="ready"</code>。初次模型加载、文件哈希校验和 NPU 权重缓存需要时间；尚未就绪时等待后重试。运行时从本地目录离线加载，不依赖原测试服务器。</p><p>如需鉴权，在创建容器时自行增加 <code>-e DECISION_API_KEY=自己的密钥</code>；后续受保护接口请求增加 <code>Authorization: Bearer 密钥</code>。健康接口保持可探测。不要将实际密钥填入要转交的文档。</p></section>
<section id="api"><h2>6. 在局域网调用</h2><p>在调用方电脑上执行。将 RK_IP 改成接收方 RK 机器自己的 IP；如果修改了映射端口，也同步修改请求地址。</p>'''
body += code('RK_IP=192.168.1.100  # 示例地址，替换为对方 RK 机器的实际 IP\ncurl --fail --max-time 10 "http://${RK_IP}:8900/health"\ncurl --fail --max-time 120 "http://${RK_IP}:8900/v1/systemone" \\\n  -H "Content-Type: application/json" \\\n  --data-binary @- <<\'JSON\'\n' + files['app/request.json'].strip() + '\nJSON')
body += '''<p>返回 <code>answers.kind.choice</code> 及各选项概率。这个请求应选择 <code>code</code>；原测试机 code 概率约 99.92%。这是对输入任务分类，响应不会生成 Python 函数正文。接口保持原 d3 请求格式：<code>state</code> 是待判定内容，<code>questions</code> 定义问题与候选项。</p></section>
<section id="verify"><h2>7. 在自己的机器复现整体速度与输出差异</h2><p>为避免与在线服务争用资源，先停自己的服务，确保其他重任务空闲。以下测试不会修改模型，只在新目录写入结果。测试失败时先查看日志，再决定是否恢复服务。</p>'''
body += code('sudo docker stop d3-edge-rknn\nexport BENCH_DIR="$D3_HOME/checks/model-$(date +%Y%m%d-%H%M%S)"\nmkdir -p "$BENCH_DIR"\nsudo docker run --rm --init --cpuset-cpus 4-7 \\\n  --memory 8g --memory-swap 8g --device /dev/dri \\\n  --mount "type=bind,source=$D3_HOME/data/models,target=/models,readonly" \\\n  --mount "type=bind,source=$BENCH_DIR,target=/work/logs" \\\n  d3-edge-rknn:1.0 /work/benchmark.py \\\n  --model /models/d3-edge --selection all --repeats 3\n\n# 检查 $BENCH_DIR/benchmark-all.json 后恢复服务\nsudo docker start d3-edge-rknn\n# 等加载完成，再检查 /health 和实际调用')
body += '''<p>基准在同一进程、4 个大核及 4 个 Torch 线程下，分别运行优化 CPU 和 RKNN。每个用例 / 后端先预热一次，再计时三次取中位数，覆盖 5 个用例、12 个问题，输入在附录的固定样本 JSON 中。</p>
<table><tr><th>字段 / 指标</th><th>验收含义</th></tr><tr><td>顶层 passed</td><td>必须为 true；执行失败或 false 均不能视为通过</td></tr><tr><td>cases.*.npu.choices_unchanged</td><td>分类选项必须与同轮优化 CPU 一致</td></tr><tr><td>max_probability_or_score_delta</td><td>同名概率 / score 取最大绝对差，排除派生 confidence、legend；要求 ≤0.01</td></tr><tr><td>speedup</td><td>CPU 三次中位数 ÷ RKNN 三次中位数；不包含 HTTP、模型加载和首次缓存建立</td></tr><tr><td>cpu_fallback_calls</td><td>记录回退次数；大输入回退 CPU 属于预期边界</td></tr></table>
<p>注意输出差异针对优化 CPU 基线，不是原始 BF16 CPU。模型内提速不能直接当作 HTTP 提速；若要测 HTTP，需要对相同输入、预热和负载条件下的 CPU / RKNN 接口分别计时。新机结果不必与历史数字逐位相同，但选项和差值必须通过验收。</p></section>
<section id="ops"><h2>8. 日常维护与配置变更</h2>'''
body += code('sudo docker ps --filter name=d3-edge-rknn\nsudo docker logs --tail 80 d3-edge-rknn\nsudo docker stop d3-edge-rknn\nsudo docker start d3-edge-rknn\n\n# 改端口、挂载、密钥或换新镜像时，重建自己的容器\nsudo docker stop d3-edge-rknn\nsudo docker rm d3-edge-rknn\n# 重新执行第 5 步 docker run，按需修改参数')
body += '''<p>删除容器不会删除绑定在本机 <code>data/models/</code> 的模型；卸载时先停止、删除自己的容器，再按需删除镜像和工作目录。不要执行全局 docker 清理命令，以免影响其他业务。</p></section>
<section id="trouble"><h2>9. 常见故障</h2><table><tr><th>现象</th><th>检查与处理</th></tr><tr><td>缺少 /dev/dri 或 rknn_* 报错</td><td>核对 RK3576、64 位系统、板厂 NPU 驱动；基线为驱动 0.9.8 + Runtime 2.5.0。先通过真实算子测试。容器不会补齐主机内核驱动。</td></tr><tr><td>Docker 权限错误</td><td>使用本文 sudo docker 命令，确认 Docker 服务已启动。</td></tr><tr><td>镜像 / wheel / SDK 下载失败</td><td>查看失败域名，检查 DNS / 代理和磁盘后重跑构建。不要任意替换为最新版依赖或跳过哈希。</td></tr><tr><td>模型下载失败</td><td>修复网络后重跑独立下载步骤，保留 data 目录。可自行配置可信 HF_ENDPOINT，但应保持相同模型版本和文件哈希校验。</td></tr><tr><td>端口占用或容器重名</td><td>改用空闲宿主端口或容器名；-p 的右侧仍为容器 8900。同步修改调用和维护命令。</td></tr><tr><td>健康接口未就绪</td><td>检查 docker logs、磁盘和内存；等模型加载 / NPU 缓存完成。容器 running 不等于模型 ready。</td></tr><tr><td>启动后退出或频繁重启</td><td>检查 OOM、模型哈希不符、驱动报错。先 docker stop，再排查；保留日志。</td></tr><tr><td>速度与历史差距大</td><td>核对大核 4–7、温度降频、后台负载和输入长度。超过 512 矩阵行会回退 CPU；不要拿矩阵微基准倍数代替完整请求倍数。</td></tr><tr><td>选项不一致或概率差超标</td><td>视为验证失败，保留 benchmark-all.json；核对模型版本、依赖及 SDK / 驱动组合后再分析。</td></tr></table></section>
<section id="appendix"><h2>附录：完整适配源码与固定测试样本</h2><p>这些文件全部内置在本 HTML。可以逐个展开复制；也可保存上方初始化脚本，运行后生成完全相同的目录。初始化不会下载模型或 SDK，实际安装由接收方按前文执行。</p>'''
body += '<details><summary>prepare_d3_sources.py（完整初始化脚本）</summary>' + code(bootstrap) + '</details>'
body += source_details
body += '''</section><section><h2>来源与版本</h2><p><a href="https://huggingface.co/vllm-sr/d3-edge/tree/20c745653400d191ecacd99c4eafc0c5900dc15e">模型固定版本</a> · <a href="https://github.com/airockchip/rknn-toolkit2/tree/2c723177a94fac65cbc036d722907ddbc91980cf">官方 RKNN 固定提交</a>。模型许可随下载文件保留；SDK 下载工具同时保留官方 LICENSE，位于镜像 /work/sdk/LICENSE。</p><p>本文仅描述 RK3576 已测方案和接收方的安装研究流程，不宣称其他 RK 芯片或新系统已经验证通过。</p></section><footer>2026-10-17 · 独立交付文档 · 无外部脚本、字体或样式依赖</footer>'''
style = '''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f6f4;color:#17343c;font:16px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}main{max-width:1060px;margin:auto;padding:38px 24px}header,section{background:white;border:1px solid #dbe5e4;border-radius:14px;padding:30px;margin-bottom:24px;scroll-margin-top:18px}h1{font-size:34px;line-height:1.3;margin-top:10px}h2{font-size:25px;margin:0 0 18px}p{margin:0 0 16px}a{color:#087e78}nav{display:flex;flex-wrap:wrap;gap:12px 20px}.eyebrow,.note,footer{color:#60757a;font-size:13px}code{font:13px/1.7 ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere}pre{padding:18px;background:#183139;color:#e4f2ed;overflow:auto;border-radius:9px;white-space:pre}pre code{overflow-wrap:normal}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;text-align:left;vertical-align:top;border-bottom:1px solid #dbe5e4}th{background:#eef4f1}details{margin:14px 0}summary{cursor:pointer;color:#087e78;overflow-wrap:anywhere}details pre{max-height:550px}button{background:#087e78;color:white;border:0;border-radius:8px;padding:12px 18px;font:inherit;cursor:pointer;margin-bottom:18px}.notice{border-left:4px solid #b57e35;background:#fbf2e4;padding:17px;font-size:14px}footer{text-align:center;padding:10px}@media(max-width:600px){main{padding:16px 12px}header,section{padding:22px 16px}h1{font-size:27px}h2{font-size:22px}body{font-size:15px}th,td{padding:8px;overflow-wrap:anywhere}button{max-width:100%;overflow-wrap:anywhere}}@media print{body{background:white}header,section{border:0;padding:10px}button,nav{display:none}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:inherit}details:not([open]){display:none}}'''
payload = json.dumps({'files': files, 'bootstrap': bootstrap}, ensure_ascii=False).replace('<', '\\u003c')
script = '''const payload=JSON.parse(document.getElementById('source-data').textContent);document.getElementById('download-init').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([payload.bootstrap],{type:'text/x-python;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='prepare_d3_sources.py';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});'''
OUT.write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>d3-edge RK3576 从零安装与使用手册</title><link rel="icon" href="data:,"><style>' + style + '</style></head><body><main>' + body + '</main><script id="source-data" type="application/json">' + payload + '</script><script>' + script + '</script></body></html>', encoding='utf-8')
print(f'{OUT}: {OUT.stat().st_size} bytes, {len(files)} embedded source files')
