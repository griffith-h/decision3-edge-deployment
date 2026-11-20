#!/usr/bin/env python3
"""Build an AI-collaboration story using historical user prompts and measured results."""
import argparse
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'd3-edge-inference-engine-optimization.html'
MD = ROOT / 'd3-edge-inference-engine-optimization.md'

# key, title, technique, AI work, effect, prompt IDs, featured prompt, measurement scope
STAGES = [
 ('explore','先让 AI 找加速路线','NPU 加速评估','查官方工具、做转换、上板测试，再检查输出是否可用。','RKLLM 能运行，但输出验证没通过。保留原服务，换路线。',['P01','P02','P03','P04'],'P04','优化起点 · 已有可用 CPU 服务'),
 ('hybrid','工具不合适，就只加速合适的部分','CPU / NPU 混合计算','大型矩阵运算交给 NPU，其余保留原计算逻辑；让 AI 做适配和对比测试。','单问题 HTTP 从 6.69 秒降到 2.82 秒。',['P05'],'P05','单问题 HTTP · 初版 3 次中位数'),
 ('profile','还有两秒，就先找两秒花在哪里','性能分析 + CPU 优化 + 线程调优','先分段计时，再处理卷积、扫描和线程竞争，最后通过真实接口复测。','三线程版部署后，100 次 HTTP 中位数约 859ms。',['P06','P07','P08','P09'],'P07','100 次 HTTP · 历史部署版本'),
 ('fusion','一秒以内后，继续减少重复工作','数据处理融合 + 独立任务并行','合并重复的数据准备，把能同时执行的任务并行起来，单独测速度与输出。','组合候选的 100 次模型内测试中位数约 665ms。',['P10','P11'],'P11','100 次模型内测试 · 当时未部署'),
 ('scan','报告指出扫描慢，就专门优化扫描','CPU 扫描优化','围绕已测瓶颈比较不同实现，做边界验证，再看完整模型的速度和输出。','候选的 100 次模型内测试中位数约 571ms。',['P12'],'P12','100 次模型内测试 · 当时未部署'),
 ('layout','反复追问：剩下的时间还能省在哪里','输出路径选择 + 负载平衡 + 减少复制','比较输出格式、任务分配、数据搬运和小块计算，更慢或输出超差的方案淘汰。','短输入候选经过约 512ms、510ms、454ms，继续推进到 442ms。',['P13','P14','P15','P16'],'P15','100 次模型内测试 · 当时未部署'),
 ('short','从逐步确认，改成有边界的连续优化','算子融合 + MLP 子图编译','给 AI 连续探索的目标，要求实测、对比和记录失败；拿到合格候选后，再授权部署。','模型内约 318ms；正式部署后，100 次 HTTP 中位数 325.27ms。',['P17','P18'],'P17','100 次 HTTP · graph-half v1 已部署'),
 ('long','短输入快了，继续解决长输入回退','按长度分块 + 加速路径扩展','依次扩展 MLP、大型投影、长序列注意力与扫描，再定位新瓶颈。','615 token 后续实验各轮终点约为 3.59 秒、2.40 秒、2.15 秒。',['P19','P20','P21','P22'],'P19','615 token · 模型内实验 · 尚未部署'),
 ('final','把目标、底线和停止条件一次说清楚','融合优化 + 并行调度 + 精度修复','先查历史失败，再自主探索；原输出门槛不变，不擅自部署，未达标也交付最优合格结果。','615 token 实验从约 2.15 秒继续推进到约 1.75 秒，最新约 1.58 秒。',['P24','P25'],'P24','615 token · 90 次模型内测试 · 尚未部署'),
]

GOALS = {
 'explore':'先看看有没有现成的 NPU 加速路线。',
 'hybrid':'不换模型，试试能正确运行的加速方式。',
 'profile':'把约两秒压到一秒以内。',
 'fusion':'继续向 200ms 探索，先找剩余瓶颈。',
 'scan':'优先处理报告中较慢的扫描。',
 'layout':'继续减少转换、复制和等待。',
 'short':'允许连续优化，拿到合格结果后再部署。',
 'long':'让不同长度的请求也能用上加速。',
 'final':'615 token 继续以一秒为目标，底线不变。',
}


def read(path):
    return json.loads((ROOT/path).read_text())


def esc(value):
    return html.escape(str(value),quote=True)


def environment():
    deployment=read('deployment.json')
    return [
        ('模型',deployment['model'],'决策模型 · 原始 12 层 · BF16 检查点权重'),
        ('用途','分类 / 判断 / 评分','根据输入返回选项概率或分数，不生成聊天回答'),
        ('设备','NanoPi R76S / RK3576','8 核 ARM：4× Cortex-A72 + 4× Cortex-A53'),
        ('内存','16GB','测试容器内存上限 8GiB'),
        ('执行方式','CPU + 板载 NPU','优化原模型的执行路径，未重新训练或裁层'),
        ('系统','ARM64 Linux + Docker','Armbian / Debian 13'),
    ]


def load():
    archive=read('ai-optimization-prompts.json')
    prompts={p['id']:p for p in archive['prompts']}
    assert len(prompts)==25 and all(p['kind'].startswith('sent_input') for p in prompts.values())
    rk=read('rknn-matmul-experiment/experiment-result.json')['http_verification']
    deployed=read('cpu-tail-deployment/deployment-result.json')['http']['timing']
    latest=read('bf16-pipeline-experiment/logs/summary.json')
    values={
      'explore':('6～7','s'),
      'hybrid':(f"{rk['rknn_single_seconds_median']:.2f}",'s'),
      'profile':(f"{read('optimized-deployment/deployment-result.json')['http_summary']['median']*1000:.2f}",'ms'),
      'fusion':(f"{read('bridge-parallel-experiment/experiment-result.json')['selected_single_engine_100']['median']*1000:.2f}",'ms'),
      'scan':(f"{read('scan-optimization-experiment/experiment-result.json')['selected_100']['median']*1000:.2f}",'ms'),
      'layout':(f"{read('attention-chunk-experiment/logs/final-single-scan.json')['timing']['median']*1000:.2f}",'ms'),
      'short':(f"{deployed['median']*1000:.2f}",'ms'),
      'long':('2.15','s'),
      'final':(f"{latest['candidate']['median']:.2f}",'s'),
    }
    return archive,prompts,values,rk


def prompt(p,featured=False):
    label='Prompt 原文' if p['kind']=='sent_input' else '交接指令 · 地址已脱敏'
    copy=f'<button type="button" class="copy js-only" data-copy="{p["id"]}" aria-label="复制 {p["id"]}">复制原文</button>'
    body=f'<div class="quote-head"><span>{p["id"]} / {label}</span>{copy}</div><blockquote>{esc(p["text"])}</blockquote>'
    if featured:return f'<div class="prompt featured">{body}</div>'
    return f'<details class="prompt" id="prompt-{p["id"]}"><summary><span>{p["id"]}</span>{esc(p["text"].splitlines()[0][:60])}</summary><div class="prompt-body">{body}</div></details>'


def bars(rows,maximum):
    return ''.join(f'<div class="story-bar"><div>{esc(label)}<small>{esc(scope)}</small></div><div class="story-track"><i style="width:{value/maximum*100:.4f}%"></i></div><b>{value:,.2f}<small>ms</small></b></div>' for label,value,scope in rows)


def measurements():
    return [
        ('短输入性能','同一请求 × 100 次','自建中文分类请求；实际输入 115 token。325.27ms 是重复计时中位数，不是 100 道题的平均值。','cpu-tail-deployment/request.json','cpu-tail-deployment/production-http.json'),
        ('长输入性能','同一请求 × 90 次','构造样例：alpha 重复 512 次，连同问题格式化后为 615 token。三轮交错对照，各 30 对；候选中位数 1.58s。','fusion-pipeline-experiment/logs/references.json','bf16-pipeline-experiment/logs/formal-1.json'),
        ('长输入候选输出回归','34 条固定请求','自建回归集，覆盖多长度、多问题与图片；对照冻结 original current 输出，无人工业务标签。','fusion-pipeline-experiment/make_references.py','bf16-pipeline-experiment/logs/regression-native.json'),
    ]


def build(archive,prompts,values,rk):
    setup=''.join(f'<article><span>{esc(label)}</span><h3>{esc(value)}</h3><p>{esc(note)}</p></article>' for label,value,note in environment())
    tests=''.join(f'<article><span>{esc(label)}</span><h3>{esc(value)}</h3><p>{esc(note)}</p><p><a href="{sample}">样例 / 构造代码</a> · <a href="{result}">原始记录</a></p></article>' for label,value,note,sample,result in measurements())
    deployment=read('deployment.json')
    cards=[]
    for i,(key,title,tech,work,effect,ids,primary,scope) in enumerate(STAGES,1):
        value,unit=values[key]
        refs=' · '.join(f'<a href="#prompt-{pid}">{pid}</a>' for pid in ids)
        cards.append(f'<article class="story-step" id="step-{key}"><span class="step-index">{i:02}</span><div><div class="step-head"><div><span class="tech">{esc(tech)}</span><h3>{esc(title)}</h3></div><div class="step-number"><strong>{value}<small>{unit}</small></strong><span>{esc(scope)}</span></div></div><div class="work"><span>我提目标</span><p>{esc(GOALS[key])}</p></div><div class="work"><span>Codex 给方案</span><p>{esc(work)}</p></div><div class="work effect"><span>确认后推进</span><p>{esc(effect)}</p></div><details class="step-prompt"><summary>查看本轮 Prompt · {len(ids)} 条</summary>{prompt(prompts[primary],True)}<div class="step-refs">本轮完整指令：{refs}</div></details></div></article>')
    http=[('优化 CPU',rk['cpu_single_seconds']*1000,'单次抽查'),('初版 CPU + RKNN',rk['rknn_single_seconds_median']*1000,'3 次中位数'),('三线程部署版',float(values['profile'][0]),'100 次中位数'),('graph-half v1',float(values['short'][0]),'100 次中位数')]
    method=[('融合与并行',float(values['fusion'][0]),'模型内测试'),('扫描优化',float(values['scan'][0]),'模型内测试'),('减少复制与分块',float(values['layout'][0]),'模型内测试'),('MLP 子图候选',read('cpu-tail-experiment/logs/formal-graph-single.json')['timing']['median']*1000,'模型内测试')]
    data=json.dumps({'prompts':[{'id':p['id'],'text':p['text']} for p in archive['prompts']],'http':http,'method':method},ensure_ascii=False).replace('<','\\u003c')
    template=r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="icon" href="data:,"><meta name="description" content="NanoPi R76S / RK3576、16GB 内存上的 d3-edge 决策模型优化复盘：从6～7秒到短输入325ms、长输入实验1.58秒，附25条当时发送给Codex的指令。"><title>d3-edge 决策模型：6～7 秒到 300ms～2s 的 AI 协作优化</title><style>@@CSS@@</style></head><body>
<a class="skip" href="#journey">跳到优化过程</a><header class="top"><div class="wrap top-inner"><a class="brand" href="#">AI × ENGINEERING / 真实复盘</a><nav aria-label="文章导航"><a href="#progress">结果路线</a><a href="#journey">优化过程</a><a href="#prompts">全部 Prompt</a></nav></div></header>
<main class="wrap"><section class="intro"><div class="eyebrow">RK3576 / d3-edge / 一次 AI 结对优化的过程</div><div class="hero-grid"><div><h1>6～7 秒，<br>一步步推到<br><em>300ms～2s。</em></h1><p class="lead">在 RK3576 上优化 d3-edge 决策模型。<br>从目标、方案到实测结果，复盘整个协作过程。</p></div><div class="hero-note"><strong>模型没换。<br>改变的是推理执行方式。</strong>我提出目标和反馈，Codex 分析瓶颈、给出下一步建议。我确认或补充后，Codex 继续实现、测试和记录。</div></div>
<div class="meta"><span class="chip">原权重 / 原 12 层</span><span class="chip">未训练 / 未蒸馏</span><span class="chip">25 条留存输入</span><span class="chip">数据截止 2026-10-18</span></div><div class="hero-metrics"><article class="metric"><div class="status">已部署 · 短文本</div><div class="number">@@SHORT@@<small>ms</small></div><p>115 token · 100 次 HTTP 中位数</p></article><article class="metric lab"><div class="status">实验候选 · 尚未部署</div><div class="number">@@LONG@@<small>s</small></div><p>615 token · 90 次模型内测试中位数</p></article></div><p class="scope">“300ms～2s”概括这两个代表性场景，不是所有请求的保证。图片、更长输入、冷启动和并发另算。</p></section>

<section class="section test-setup" id="setup"><h2>用什么模型，跑在什么机器上</h2><div class="setup-grid">@@SETUP@@</div><details class="more"><summary>软件版本与测试设置</summary><div class="setup-details"><p>PyTorch @@TORCH@@ · Transformers @@TRANSFORMERS@@ · RKNN Runtime 2.5.0 · NPU 驱动 0.9.8。</p><p>测试固定使用 CPU 4–7；已部署版 Torch / OpenMP 为 3 线程，后续长输入实验的扫描另用 4 线程。以上性能为模型加载后的暖态、单请求测量。</p><p>模型固定 revision：<code>@@REVISION@@</code>。</p><p>资料：<a href="https://huggingface.co/vllm-sr/d3-edge">模型主页</a> · <a href="cpu-tail-deployment/README.md">环境与部署记录</a></p></div></details></section>

<section class="section" id="measurement"><h2>用什么测，别人怎么复现</h2><p class="section-lead">本次使用仓库自建的固定请求与回归集，未重跑官方模型卡列出的 Jev Decision Index 等基准。</p><div class="setup-grid">@@TESTS@@</div><p class="scope">性能样本数 ≠ 不同题目数。34 例通过表示已测输出符合数值门槛，不表示业务准确率不变。</p><details class="more"><summary>展开测试口径与复现入口</summary><div class="setup-details"><p>34 例 = 历史 9 例 + 1 例多问题请求 + 12 例替换词的 615 token 输入 + 12 例额外短请求；覆盖选择、判断与评分。参照来自单独重算的 original current 路径，参照答案只用于对比。</p><p>长输入候选要求 choice 不变，概率 / score / noul 最大绝对差 ≤ 0.01；confidence 单独报告，实测最大差约 0.01415。短输入部署另验证了 19 个 HTTP 场景 / 26 个问题，与上述 34 例分开。</p><p>正式长输入每模式先预热 3 次，再做三轮各 30 对交错计时；保留全部正式样本，P95 使用 nearest-rank。约 6.69s 的早期 CPU 数字仅为单次 HTTP 抽查。</p><p><a href="article-reproduction.md">完整复现说明：请求、统计、命令与依赖</a> · <a href="cpu-tail-deployment/README.md">已部署版复测</a> · <a href="bf16-pipeline-experiment/REPRODUCTION.md">长输入实验复现</a> · <a href="https://huggingface.co/vllm-sr/d3-edge/blob/20c745653400d191ecacd99c4eafc0c5900dc15e/README.md#evaluation">官方模型卡的评测</a></p><p>仓库留有源码、固定请求和原始输出；新机器还需下载模型与 SDK、构建 ARM 库和 RKNN 图，并准备历史依赖。完整克隆并非一键复现；离线单个 HTML 也不包含这些数据文件。</p></div></details></section>

<section class="section"><h2>我提目标，Codex 给方案，确认后继续</h2><div class="loop"><b>我提目标</b><span>→</span><b>Codex 分析与建议</b><span>→</span><b>我确认或补充</b><span>→</span><b>Codex 实现与验证</b><span>→</span><b>根据结果再继续</b></div><p class="scope">部分指令采用了 Codex 的建议或整理内容，下面保留当时发送的原文。</p></section>

<section class="section" id="progress"><h2>先看结果走过的路</h2><p class="section-lead">接口测试与模型内测试分开。不同轮次展示演进，不直接相除计算整体提速倍数。</p><div class="duo"><div class="panel"><h3>真实接口：6～7 秒 → 约 325ms</h3><p class="chart-note">短文本 HTTP · 中位数或单次抽查</p>@@HTTP@@<p class="scope">条形从 0 开始，共同上限 7000ms；早期样本量较小。</p></div><div class="panel method"><h3>中间候选：持续减少模型内耗时</h3><p class="chart-note">115 token · 每阶段 100 次暖态测试</p>@@METHOD@@<p class="scope">条形从 0 开始，共同上限 700ms；候选当时未直接替换服务。</p></div></div></section>

<section class="section" id="journey"><h2>每一步怎么推进，快了多少</h2><p class="section-lead">看目标、方案与结果；需要时再展开 Prompt。</p>@@CARDS@@</section>

<section class="section" id="prompts"><h2>Prompt 原文，完整保留</h2><details class="archive-wrap" id="prompt-archive"><summary>展开 25 条 Prompt 的完整时间线</summary><div class="toolbar"><span class="scope">展开单条看原文，也可以全部展开。</span><button type="button" class="plain-button js-only" id="expand-prompts">展开全部 Prompt</button></div><div class="archive">@@PROMPTS@@</div><p class="scope">P13 的“469ms”是当时扫描之外的剩余项，不是整网耗时。P23 的旧误差记录是当时状态，不代表最新结论。</p></details></section>

<p class="scope closing-note">短输入已部署，长输入仍是实验；200ms 与长输入一秒目标尚未达成。保留原权重与层数，已测输出做了回归，不等于业务准确率评测。</p></main>
<footer class="footer"><div class="wrap footer-row"><span>一次真实的 AI 协作优化复盘 · 2026-10-19 改稿</span><span><a href="d3-edge-inference-engine-optimization.md">Markdown 原稿</a> · <a href="ai-optimization-prompts.json">Prompt 归档</a> · <a href="WORK_STATUS.md">状态与数据</a> · <a href="cpu-tail-deployment/deployment-result.json">部署结果</a> · <a href="bf16-pipeline-experiment/logs/summary.json">实验结果</a></span><button class="plain-button js-only" id="print-page" type="button">打印 / 保存 PDF</button><span id="copy-status" role="status" aria-live="polite"></span></div></footer>
<script type="application/json" id="article-data">@@DATA@@</script><script>
(()=>{'use strict';document.documentElement.classList.add('js');const data=JSON.parse(document.getElementById('article-data').textContent),prompts=new Map(data.prompts.map(p=>[p.id,p.text]));
async function copy(text){if(navigator.clipboard){try{await navigator.clipboard.writeText(text);return}catch{}}const box=document.createElement('textarea');box.value=text;box.style.position='fixed';box.style.opacity='0';document.body.append(box);box.select();const ok=document.execCommand('copy');box.remove();if(!ok)throw Error('copy failed')}
document.querySelectorAll('[data-copy]').forEach(button=>button.addEventListener('click',async()=>{try{await copy(prompts.get(button.dataset.copy));button.textContent='已复制';document.getElementById('copy-status').textContent=button.dataset.copy+' 原文已复制'}catch{document.getElementById('copy-status').textContent='请手动选择并复制 Prompt 原文'}}));
const list=[...document.querySelectorAll('.archive>details')],button=document.getElementById('expand-prompts');function label(){const all=list.every(d=>d.open);button.textContent=all?'收起全部 Prompt':'展开全部 Prompt';button.setAttribute('aria-expanded',String(all))}button.addEventListener('click',()=>{const all=list.every(d=>d.open);list.forEach(d=>d.open=!all);label()});list.forEach(d=>d.addEventListener('toggle',label));label();document.querySelectorAll('.step-refs a').forEach(a=>a.addEventListener('click',()=>{const d=document.getElementById(a.hash.slice(1));if(d){document.getElementById('prompt-archive').open=true;d.open=true}}));document.getElementById('print-page').addEventListener('click',()=>window.print());})();
</script></body></html>'''
    replacements={'CSS':(ROOT/'optimization-article.css').read_text(),'SHORT':values['short'][0],'LONG':values['final'][0],
      'SETUP':setup,'TESTS':tests,'TORCH':esc(deployment['torch_version']),'TRANSFORMERS':esc(deployment['transformers_version']),'REVISION':esc(deployment['model_revision']),
      'HTTP':bars(http,7000),'METHOD':bars(method,700),'CARDS':''.join(cards),
      'PROMPTS':''.join(prompt(p) for p in archive['prompts']),'DATA':data}
    for key,value in replacements.items():template=template.replace('@@'+key+'@@',value)
    assert '@@' not in template
    return template


def markdown(archive,prompts,values):
    deployment=read('deployment.json')
    setup='| 项目 | 本次使用 |\n| --- | --- |\n'+'\n'.join(f'| {label} | {value}；{note} |' for label,value,note in environment())
    parts=['# 我和 Codex，把 RK3576 上的 d3-edge 决策模型从 6～7 秒优化到 300ms～2s',
      '> 一次真实的 AI 协作优化复盘。数据截止 2026-10-18；已发送指令核对与改稿日期为 2026-10-19。',
      '最初，手里的 d3-edge 可用服务一次请求约要 6～7 秒。我的目标很直接：不换模型，能不能让它快到真正好用？',
      '推进方式是多轮协作：我提出目标和反馈，Codex 分析瓶颈、提出路线与下一步指令；我确认或补充后，Codex 继续实现、测试和记录。部分指令采用了 Codex 的建议或整理内容，以下保留当时发送的原文。',
      '| 场景 | 结果 | 状态 |\n| --- | --- | --- |\n| 短文本，115 token | HTTP 中位数 325.27ms，100 次 | graph-half v1 已部署 |\n| 较长文本，615 token | 模型内中位数 1579.62ms，90 次 | 实验候选，尚未部署 |',
      '“300ms～2s”概括这两个代表性场景，不是所有请求的保证。图片、更长输入、冷启动和并发另算。模型权重与 12 层结构保留；已测输出回归不等于业务准确率评测。',
      '## 模型与测试环境',setup,
      f'软件：PyTorch {deployment["torch_version"]}、Transformers {deployment["transformers_version"]}、RKNN Runtime 2.5.0、NPU 驱动 0.9.8。',
      '测试固定使用 CPU 4–7；已部署版 Torch / OpenMP 为 3 线程，后续长输入实验的扫描另用 4 线程。性能为模型加载后的暖态、单请求测量。',
      f'模型固定 revision：`{deployment["model_revision"]}`。资料：[模型主页](https://huggingface.co/vllm-sr/d3-edge) · [环境与部署记录](cpu-tail-deployment/README.md)。',
      '## 用什么测，别人怎么复现',
      '本次使用仓库自建的固定请求与回归集，未重跑[官方模型卡列出的 Jev Decision Index 等基准](https://huggingface.co/vllm-sr/d3-edge/blob/20c745653400d191ecacd99c4eafc0c5900dc15e/README.md#evaluation)。',
      '| 测什么 | 怎么测 | 样例与证据 |\n| --- | --- | --- |\n'+'\n'.join(f'| {label}：{value} | {note} | [样例 / 构造代码]({sample}) · [原始记录]({result}) |' for label,value,note,sample,result in measurements()),
      '34 例由历史 9 例、1 例多问题请求、12 例替换词的 615 token 输入和 12 例额外短请求组成，覆盖选择、判断与评分。参照来自单独重算的 original current 路径，参照答案只用于对比。门槛为 choice 不变，概率 / score / noul 最大绝对差 ≤ 0.01；confidence 单列，实测最大差约 0.01415。已部署短输入版另验证 19 个 HTTP 场景 / 26 个问题。',
      '性能样本数不等于不同题目数。长输入每模式预热 3 次，三轮各 30 对交错计时；保留全部正式样本，P95 使用 nearest-rank。约 6.69s 的早期 CPU 结果仅为单次 HTTP 抽查。以上不等于真实业务集的平均性能或准确率评测。',
      '[完整复现说明](article-reproduction.md)包含固定请求、原始结果重算、复测命令和依赖准备。模型、SDK、ARM 库与 RKNN 图须另行下载或构建，克隆仓库并非一键复现；单独 HTML 也不包含链接的数据文件。',
      '## 优化是怎么一步步推进的']
    for i,(key,title,tech,work,effect,ids,primary,scope) in enumerate(STAGES,1):
        parts += [f'### {i:02}. {title}',f'**用了什么：** {tech}。',f'**我提目标：** {GOALS[key]}',f'**Codex 给方案：** {work}',f'**确认后推进：** {effect}',f'**本阶段代表结果：** {"".join(values[key])}；{scope}。','本轮 Prompt：'+'、'.join(ids)+'（见下文时间线）。']
    parts += ['## Prompt 原文：25 条完整时间线', '以下保留优化与部署过程中当时发送的文字，交接指令中的机器地址已脱敏。', 'P13 的469ms是当时扫描之外的剩余项，不是整网耗时；P23 的旧误差是当时状态，不代表最新结论。']
    for p in archive['prompts']:
        label='交接指令，地址已脱敏' if p['kind']=='sent_input_redacted' else 'Prompt 原文'
        parts += [f'### {p["id"]} · {label}','\n'.join('>'+(' '+line if line else '') for line in p['text'].splitlines())]
    parts += ['## 发布时只说已经做到的部分','- 短输入 graph-half v1 已部署；最新长输入候选仍是实验。','- 200ms 和长输入一秒目标没有都达成，不把目标写成结果。','- HTTP 与模型内测试分别标注，不混合计算整体提速倍数。','- 原始模型权重和层数保留，执行精度与实现有变化。没有标注业务数据集，不写“准确率完全无损”。',
      '原始资料：[Prompt 归档](ai-optimization-prompts.json) · [当前状态](WORK_STATUS.md) · [部署结果](cpu-tail-deployment/deployment-result.json) · [最新实验](bf16-pipeline-experiment/logs/summary.json) · [图解 HTML](d3-edge-inference-engine-optimization.html)']
    return '\n\n'.join(parts)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--write-markdown',action='store_true',help='Explicitly regenerate the Markdown draft too')
    args=parser.parse_args()
    assert not(args.check and args.write_markdown)
    archive,prompts,values,rk=load();result=build(archive,prompts,values,rk)
    if args.check:
        assert OUT.read_text()==result,'HTML stale; run python3 build_optimization_article.py'
        print('PASS: HTML matches 25 verified prompts and source results')
    else:
        OUT.write_text(result);print(f'Wrote {OUT.name}: {len(result.encode()):,} bytes')
    if args.write_markdown:
        MD.write_text(markdown(archive,prompts,values));print(f'Wrote {MD.name}')


if __name__=='__main__':main()
