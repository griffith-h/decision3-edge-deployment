#!/usr/bin/env python3
"""Generate an offline Chinese report from retained experiment evidence."""
import datetime
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'd3-edge-rk3576-report.html'
def read(path):
    return json.loads((ROOT / path).read_text())
def esc(value):
    return html.escape(str(value), quote=True)

rk = read('rknn-matmul-experiment/experiment-result.json')
llm = read('rkllm-experiment/experiment-result.json')
shutdown = read('shutdown-verification.json')
activation = read('service-activation-verification.json') if (ROOT/'service-activation-verification.json').exists() else None
bench = read('rknn-matmul-experiment/logs/benchmark-all.json')
api = read('rknn-matmul-experiment/logs/api-verification.json')
optim = read('optimization-results.json')
labels = {'single':'中文编程 · 单问题','chinese-0':'中文编程 · 三问题','chinese-1':'中文聊天 · 三问题','english':'英文编程 · 三问题','image':'小图片 · 两问题'}
model_rows = [{'name':labels[key], 'cpu':x['cpu_seconds'], 'fast':x['rknn_seconds'],
               'speedup':x['speedup'], 'delta':x['max_probability_or_score_delta']}
              for key,x in rk['model_benchmarks']['cases'].items()]
http_rows = [{'name':'中文编程 · 单问题', 'cpu':api['cpu_preservation']['seconds'],
              'fast':api['single_npu_seconds_median'], 'speedup':api['single_http_speedup'],
              'delta':max(x['max_delta_vs_cpu'] for x in api['requests'] if x['case']=='single')}]
history_rows = []
oldsingle = read('restart-results.json')
new_single = read('optimized-production-result.json')
history_rows.append({'name':'中文编程 · 单问题','cpu':oldsingle['request_seconds'],'fast':new_single['seconds'],
                     'speedup':oldsingle['request_seconds']/new_single['seconds'],'delta':None})
for x,label in zip(optim,['中文编程 · 三问题','中文聊天 · 三问题','英文编程 · 三问题','小图片 · 两问题']):
    history_rows.append({'name':label,'cpu':x['baseline_seconds'],'fast':x['seconds'],
                         'speedup':x['speedup'],'delta':x['max_probability_or_score_delta']})
datasets = {
 'http':{'title':'实际 HTTP 服务','before':'优化 CPU / 8899','after':'CPU + RKNN / 8900','rows':http_rows,
         'method':'CPU 服务复测 1 次；RKNN 服务连续请求 3 次取中位数。完整 HTTP 请求计时，服务已加载并预热；不含服务启动。两个服务按顺序测量。'},
 'model':{'title':'模型内成对基准','before':'优化 CPU','after':'CPU + RKNN','rows':model_rows,
          'method':'同一进程、CPU 核心 4–7、4 个 Torch 线程；每个后端/用例先预热，再测量 3 次取中位数。计时含请求准备与原始 d3 输出处理，不含 HTTP、模型加载及首次 NPU 缓存建立。'},
 'history':{'title':'早期 CPU 优化','before':'原始 BF16 CPU','after':'FP32 算子优化 CPU','rows':history_rows,
            'method':'早期 HTTP 样本的单次测量，用于回顾 CPU 优化收益。与本轮 RKNN 基准的计时轮次、运行阶段不同，请分别解读，不能直接拼接为整体加速比。'}
}
def perf_table(rows):
    return '<div class="table-wrap"><table><thead><tr><th>请求用例</th><th>基线 / 秒</th><th>优化后 / 秒</th><th>提速</th></tr></thead><tbody>' + ''.join(
        f'<tr><td>{esc(r["name"])}</td><td class="num">{r["cpu"]:.2f}</td><td class="num accent">{r["fast"]:.2f}</td><td class="num">{r["speedup"]:.2f}×</td></tr>' for r in rows) + '</tbody></table></div>'
def bars(rows):
    maxval=max(r['cpu'] for r in rows)
    return ''.join(f'''<div class="bar-row"><div class="bar-title">{esc(r['name'])}<span>{r['speedup']:.2f}×</span></div>
      <div class="bar-track"><div class="bar cpu" style="width:{r['cpu']/maxval*100:.2f}%" title="基线 {r['cpu']:.3f} 秒"></div><b>{r['cpu']:.2f}s</b></div>
      <div class="bar-track"><div class="bar fast" style="width:{r['fast']/maxval*100:.2f}%" title="优化后 {r['fast']:.3f} 秒"></div><b>{r['fast']:.2f}s</b></div></div>''' for r in rows)
panels=''.join(f'''<section class="perf-panel" id="panel-{key}" role="tabpanel" aria-labelledby="tab-{key}" {'hidden' if key!='http' else ''}>
<h3>{d['title']}</h3><p class="method">{d['method']}</p><div class="legend"><span><i class="cpu"></i>{d['before']}</span><span><i class="fast"></i>{d['after']}</span></div>
<div class="bar-chart" aria-label="耗时对比，横条越短耗时越少">{bars(d['rows'])}</div>{perf_table(d['rows'])}</section>''' for key,d in datasets.items())
accuracy_rows=''.join(f'''<tr><td>{esc(r['name'])}</td><td class="num">{r['delta']:.6f}</td>
<td><div class="delta-track" title="使用容差预算的 {r['delta']/0.01*100:.1f}%"><i style="width:{r['delta']/0.01*100:.2f}%"></i></div></td><td><span class="badge pass">选择一致 · 通过</span></td></tr>''' for r in model_rows)
stopped_rows=''.join(f'<tr><td><code>{esc(c["name"])}</code></td><td>{"8899" if c["name"]=="decision3-edge" else "8900"}</td><td><span class="badge neutral">已停止</span></td><td><code>{esc(c["restart_policy"])}</code></td></tr>' for c in shutdown['containers'])
time_local=datetime.datetime.fromisoformat(shutdown['verified_at_utc']).astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8')
status_title='报告已更新 · 等待启动核验'
status_detail='RKNN 8900 待启动；CPU 8899 保持停止'
if activation:
    assert activation['passed'] and activation['health']['status']=='ready'
    time_local=datetime.datetime.fromisoformat(activation['verified_at_utc']).astimezone(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S UTC+8')
    status_title='RKNN 服务已启动并通过调用核验'
    status_detail='RKNN 8900 在线 · CPU 8899 保持停止'
    stopped_rows=''.join(f'<tr><td><code>{esc(c["name"])}</code></td><td>{"8899" if c["name"]=="decision3-edge" else "8900"}</td><td><span class="badge {"pass" if c["running"] else "neutral"}">{"运行中" if c["running"] else "已停止"}</span></td><td><code>{esc(c["restart_policy"])}</code></td></tr>' for c in activation['containers'])
sources={
 'shutdown-verification.json':shutdown,
 'rknn-matmul-experiment/experiment-result.json':rk,
 'rknn-matmul-experiment/logs/benchmark-all.json':bench,
 'rknn-matmul-experiment/logs/api-verification.json':api,
 'rknn-matmul-experiment/logs/micro-test.json':read('rknn-matmul-experiment/logs/micro-test.json'),
 'rknn-matmul-experiment/logs/boundary-test.json':read('rknn-matmul-experiment/logs/boundary-test.json'),
 'rknn-matmul-experiment/logs/thread-diagnosis.json':read('rknn-matmul-experiment/logs/thread-diagnosis.json'),
 'rkllm-experiment/experiment-result.json':llm,
 'rkllm-experiment/logs/hidden-diagnosis.json':read('rkllm-experiment/logs/hidden-diagnosis.json'),
 'rkllm-experiment/logs/exported-attention-metadata.json':read('rkllm-experiment/logs/exported-attention-metadata.json'),
 'optimization-results.json':optim,
 'golden-results.json':read('golden-results.json'),
 'restart-results.json':oldsingle,
 'optimized-production-result.json':new_single,
 'optimized-restart-result.json':read('optimized-restart-result.json')
}
if activation:
    sources['service-activation-verification.json']=activation
sources['rknn-matmul-experiment/sdk-manifest.json']=read('rknn-matmul-experiment/sdk-manifest.json')
conclusion_rows=''.join(f'<tr><td>{esc(r["name"])}</td><td class="num">{r["cpu"]:.3f}</td><td class="num">{r["fast"]:.3f}</td><td class="num accent">{r["speedup"]:.2f}×</td><td class="num">{r["delta"]:.8f}</td><td>一致</td></tr>' for r in model_rows)
example="curl --max-time 120 'http://192.168.1.164:8900/v1/systemone' \\\n  -H 'Content-Type: application/json' \\\n  --data-binary @- <<'JSON'\n" + json.dumps(read('request.json'),ensure_ascii=False,indent=2) + "\nJSON"
final_section='''<section class="section" id="conclusion"><div class="section-heading"><div><h2>最终结论：采用保留原计算逻辑的 RKNN 混合加速</h2><p class="sub">性能与输出差异来自 2026-10-13 的实测；服务状态另按本次启动核验记录。</p></div><span class="index">FINAL / 结论</span></div>
<p><strong>RKNN 方案在已测样本中可用。</strong>纯文本模型内成对测试提速 <strong>2.19–2.74 倍</strong>，小图片提速 <strong>1.43 倍</strong>。完整 HTTP 单问题历史实测从 6.686 秒降至三次请求中位数 2.822 秒，约 <strong>2.37 倍</strong>，耗时减少约 57.8%。HTTP 与模型内计时分别比较，不混算。</p>
<div class="table-wrap"><table><thead><tr><th>模型内成对测试</th><th>CPU / 秒</th><th>RKNN / 秒</th><th>提速</th><th>概率 / score 最大绝对差</th><th>选项</th></tr></thead><tbody>'''+conclusion_rows+'''</tbody></table></div>
<p class="note"><strong>精度含义：</strong>以优化 CPU 为基线，5 个用例、12 个问题的选项全部一致；最大概率 / score 绝对差为 0.00283136，低于约定容差 0.01。若该数值来自概率，等于约 0.283 个百分点；它不代表任务准确率下降 0.283%，也不是完整数据集准确率评测。confidence 不计入该差值。原始 CPU 到优化 CPU 的差异属于另一组基线，不能据此推导原始 CPU 到 RKNN 的端到端精度上限。</p>
<p>加速方式是把 78 个文本线性层的矩阵乘交给 NPU，保留原始非因果注意力、因果 GatedDeltaNet 扫描、视觉塔和输出 readout。使用官方原始 12 层结构及权重，<strong>没有进一步裁剪，也不需要整模型转成 RKNN 文件</strong>；启动时缓存 FP16 矩阵。超过 512 矩阵行会回退 CPU，长文本和视频尚无性能结论。</p>
<p><strong>RKLLM 整模型路线停止继续适配。</strong>本次 SDK / Runtime 1.3.1 可以转换并运行，但非因果注意力未正确生效，最大概率差 0.752699、选项仅 2/4 一致，不能作为可用替代。已写适配组件没有接入官方执行器，不构成修复；本轮不进行厂家二进制逆向。</p></section>'''
api_section='<section class="section" id="api"><div class="section-heading"><div><h2>局域网调用示例</h2><p class="sub">这是分类 / 评分接口。示例判断输入属于编程、数学还是聊天，并不生成 Python 函数正文。</p></div><span class="index">API / 8900</span></div><p>健康检查：<a href="http://192.168.1.164:8900/health"><code>http://192.168.1.164:8900/health</code></a>；推理接口：<code>POST /v1/systemone</code>。</p><pre style="overflow:auto;padding:18px;background:#183139;color:#dfebe8;border-radius:9px;font:12px/1.7 ui-monospace,monospace"><code>'+esc(example)+'</code></pre><p class="note">返回 answers.kind.choice 及各选项概率；本例预期选择 code。请求串行执行，超过原生矩阵支持范围时保留 CPU 回退路径。</p></section>'
def code_block(code):
    return '<pre style="overflow:auto;padding:18px;background:#183139;color:#dfebe8;border-radius:9px;font:12px/1.7 ui-monospace,monospace"><code>'+esc(code)+'</code></pre>'
reproduction_section='''<section class="section" id="reproduction"><div class="section-heading"><div><h2>如何复现 RKNN 提速与输出差异</h2><p class="sub">复用本机归档与当前 RK3576 板端环境；新跑的结果单独保存，不覆盖历史证据。</p></div><span class="index">REPRODUCE / 复现</span></div>
<h3>1. 固定版本和输入</h3><p>在 <code>192.168.1.164</code> 的 root shell 中执行后续板端命令。依赖为 Docker、g++、Python 3、可访问的 <code>/dev/dri</code> 和现有镜像 <code>decision3-edge:arm64-cpu</code>。镜像原测试 ID 为 <code>sha256:cd5c294eb741dd011ad31f8dbc6ab10a088123922122f06d2c29b7d89067d49d</code>；Torch 2.12.0+cpu、Transformers 5.17.0，CPU 核心 4–7、4 个 Torch 线程、8 GiB 容器内存。</p>
<p>模型固定为 <code>vllm-sr/d3-edge@20c745653400d191ecacd99c4eafc0c5900dc15e</code>，使用官方 12 层权重；每次模型加载执行完整文件哈希验证。缓存位于 <code>/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime</code>，需保留其 <code>models--vllm-sr--d3-edge/snapshots/版本号</code> 层级。板端源码在 <code>/opt/decision3-edge-matmul</code>，SDK 固定提交 <code>2c723177a94fac65cbc036d722907ddbc91980cf</code>，Runtime 2.5.0、驱动 0.9.8。脚本对 SDK 文件按 <code>sdk-manifest.json</code> 校验 SHA256。</p>
<p>复现所需文件包括 <code>matmul_bridge.cpp</code>、<code>rknn_linear.py</code>、<code>cpu_optimized_server.py</code>、<code>benchmark.py</code>、<code>micro_test.py</code>、<code>boundary_test.py</code>、<code>request-cases.json</code>、<code>golden-reference.json</code> 和 <code>sdk/</code>。它们已保存在板端和本机项目的 <code>rknn-matmul-experiment/</code>；重建镜像使用项目根目录的 <code>Dockerfile</code> 与 <code>Dockerfile.server</code>，依次构建标签 <code>decision3-edge-rk3576:torch2.12</code> 和 <code>decision3-edge:arm64-cpu</code>。</p>'''+code_block('docker image inspect decision3-edge:arm64-cpu --format \'{{.Id}}\'\nls /dev/dri\nls /opt/decision3-edge-matmul/{sdk,request-cases.json,golden-reference.json}')+'''
<h3>2. 运行完整模型与矩阵测试</h3><p>下面的复现脚本会<strong>暂时停止当前运行的 CPU / RKNN 服务</strong>以避免争用，在独立目录编译桥接库，依次执行矩阵测试、边界测试和 5 个完整模型用例。退出时尝试恢复原先运行的容器，原先停止的容器保持停止；启动完成后仍应检查健康接口。此脚本本轮已做 <code>bash -n</code> 语法检查，未重新执行整套性能测试。</p>'''+code_block('bash /opt/decision3-edge-matmul/reproduce-rknn.sh')+'''
<p>每个模型用例先预热，再对 CPU 和 RKNN 各计时 3 次取中位数；不计模型加载、NPU 权重缓存建立及 HTTP 时间。提速 = CPU 中位数 ÷ RKNN 中位数；对同一输出路径的概率 / score 取最大绝对差，排除 confidence、legend。通过标准：选项一致且最大差 ≤ 0.01。输入以固定的 <code>request-cases.json</code> 和 <code>golden-reference.json</code> 为准，不临时替换样本。</p>
<p>新结果位于 <code>/opt/decision3-edge-matmul/reproduction-时间-随机串/logs/</code>，包括 <code>benchmark-all.json</code>、<code>micro-test.json</code>、<code>boundary-test.json</code> 及控制台日志；主脚本最后打印目录、耗时、倍数、最大差和选项一致性。原始 <code>/opt/decision3-edge-matmul/logs/</code> 不覆盖。新测值受温度、后台负载及运行波动影响，不保证与报告逐位一致。</p>
<details><summary>展开完整复现脚本（HTML 内置）</summary>'''+code_block((ROOT/'rknn-matmul-experiment/reproduce-rknn.sh').read_text())+'''</details>
<h3 style="margin-top:20px">3. 启停与接口核验</h3><p>已有容器直接 start；首次创建容器才使用 start-npu-test-service.sh，后者默认配置自动重启。当前已有容器的重启策略为 no，下面命令保持该策略。</p>'''+code_block("docker start decision3-edge-matmul\ndocker inspect decision3-edge-matmul --format '{{.State.Status}} {{.State.Health.Status}}'\ncurl --fail --max-time 10 http://192.168.1.164:8900/health\n# 应返回 status=ready；若尚未加载完，等待后再检查\n# 停止 RKNN 服务：docker stop decision3-edge-matmul")+'''<p>健康接口就绪后，用下一节的完整 JSON 请求验证推理。HTTP 计时包含网络、服务处理和推理；若重新计算 HTTP 提速，需要在同等负载和预热条件下分别测 CPU 8899 与 RKNN 8900，不能把本次启动耗时与历史 CPU 耗时相除。</p></section>'''
live_details=''
if activation:
    live_details='<p class="note">本次重新启动的实际请求：'+ '；'.join(f'{esc(x["case"])} {x["seconds"]:.3f} 秒，较历史优化 CPU 输出最大差 {x["max_probability_or_score_delta_vs_historical_optimized_cpu"]:.8f}' for x in activation['requests'])+'。选项均一致；无效请求返回 422。本次只做恢复运行检查，没有重跑 CPU 计时，不据此更新历史提速倍数。</p>'
support_section=''
if (ROOT/'rkllm-d3-support/result.json').exists():
    for name in ['rkllm-d3-support/result.json','rkllm-d3-support/logs/qwen-path.json',
                 'rkllm-d3-support/logs/writer-patch-readback.json','rkllm-d3-support/logs/sdk-public-api.json']:
        sources[name]=read(name)
    support_section='''<section class="section" id="rkllm-support"><div class="section-heading"><div><h2>RKLLM 修改审查：执行器尚未接入</h2><p class="sub">后续源码与接口审查。此项审查当时没有启动服务，也没有新增 RKLLM 原生模型精度或速度结果。</p></div><span class="index">03B / NATIVE SUPPORT</span></div>
    <div class="two-column"><div><p>官方加载器原本已经选择 <code>Qwen3_5DecoderLayer</code> 的专用映射；导出架构名称 <code>custom</code> 是这套适配的实现方式。不能靠更改模型名完成非因果支持。</p><p>当前公开版本提供预编译 <code>librkllmrt.so</code>，转换器与加载器核心也以编译模块发布。公开 Runtime 接口没有自注意力 mask setter 或逐层回调，交叉注意力的 encoder mask 接口也不能替代这项自注意力修改。</p><ul class="bullet-list"><li>已写导出适配：校验原始配置哈希、保留官方 12 层排列，写入注意力与预填充约定。</li><li>已写可移植 C++ 策略：按显式层类型选择行为，完整注意力允许未来 key；检查全序列 K/V 就绪，保留线性扫描边界。</li><li>6 个 Python 单元测试通过；官方 Writer 序列化 / 读回通过；C++ 策略在 macOS 与 ARM64 板端编译和测试通过。</li></ul></div>
    <aside class="small-panel failure"><span class="badge warn">尚未完成原生 RKLLM 修复</span><h3 style="margin-top:16px">仍需执行器层面的支持</h3><ul class="bullet-list"><li>读取逐层类型数组，替换固定周期布局推导。</li><li>完整注意力保留 padding mask，允许读取同一提示词的未来 token。</li><li>在每个完整注意力层先准备全部 K/V，再按块计算查询；保留 GatedDeltaNet 因果扫描。</li><li>禁用非因果请求的旧 decoder history 复用，继续使用原始 readout。</li></ul><p>可移植策略头文件没有被官方二进制加载，也没有替换它的调度器。元数据样例不含模型权重；新增约定字段需要原生代码实现，不能当作当前 SDK 的隐藏开关。现有 Runtime 会被能力检查拒绝。</p><p><strong>下一步依赖：</strong>Rockchip 修改版 Runtime / 编译器源码与构建依赖，或实现相同语义的新 SDK。以上组件单测不等于实际 NPU 模型通过。</p></aside></div></section>'''
evidence_options=''.join(f'<option value="{esc(name)}">{esc(name)}</option>' for name in sources)
data=json.dumps({'datasets':datasets,'sources':sources},ensure_ascii=False).replace('<','\\u003c')

template=r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>d3-edge × RK3576 · 部署与加速测试报告</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 40 40'%3E%3Crect width='40' height='40' rx='10' fill='%23087e78'/%3E%3Ctext x='8' y='27' font-size='20' font-family='sans-serif' fill='white'%3Ed3%3C/text%3E%3C/svg%3E">
<style>
:root{--ink:#17343c;--muted:#60757a;--teal:#087e78;--line:#dbe5e4;--paper:#fff;--bg:#f3f6f4;--cpu:#8da0a7;--amber:#a96919;--red:#a6453d}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.65;font-size:15px}a{color:var(--teal);text-decoration:none}a:hover{text-decoration:underline}button,select{font:inherit}button{cursor:pointer}button:focus-visible,a:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid #d6a957;outline-offset:4px}
.masthead{border-bottom:1px solid var(--line);background:#fbfcfb}.masthead-inner{max-width:1180px;margin:auto;padding:17px 32px;display:flex;justify-content:space-between;align-items:center;gap:20px}.brand{font-weight:750;font-size:14px;letter-spacing:1px}.nav{display:flex;gap:22px;font-size:13px}.page{max-width:1180px;margin:auto;padding:44px 32px 70px}.eyebrow{text-transform:uppercase;font-size:12px;font-weight:700;letter-spacing:2px;color:var(--teal)}h1{font-size:43px;letter-spacing:-1.5px;line-height:1.25;margin:14px 0 19px;font-weight:750}h1 span{font-weight:500;color:#648086}h2{font-size:25px;line-height:1.35;margin:0 0 10px;letter-spacing:-.5px}h3{font-size:17px;margin:0 0 12px}p{margin:0 0 14px}.hero{display:grid;grid-template-columns:1fr 285px;gap:45px;align-items:start}.lede{color:var(--muted);max-width:725px;font-size:16px}.stamp{padding:23px;border:1px solid var(--line);border-radius:15px;background:#e9efed}.stamp .status{display:flex;align-items:center;gap:8px;font-weight:750;margin-bottom:12px}.dot{width:8px;height:8px;background:#748883;border-radius:50%}.stamp p{font-size:13px;color:var(--muted);margin-bottom:7px}.actions{display:flex;gap:10px;margin-top:18px}.button{border:1px solid var(--line);border-radius:8px;padding:8px 13px;background:white;font-size:12px;font-weight:600;color:var(--ink)}.button:hover{background:#e8f1ef}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:29px 0 27px}.metric{background:var(--paper);border:1px solid var(--line);padding:21px 23px;border-radius:13px}.metric-label{font-size:12px;color:var(--muted)}.metric-number{font-size:33px;font-weight:750;letter-spacing:-1px;margin:5px 0;color:var(--teal)}.metric-note{font-size:12px;color:var(--muted)}.callout{padding:19px 23px;border-left:4px solid var(--teal);background:#e6f1ed;border-radius:0 10px 10px 0;margin-bottom:32px}.callout p{margin:0;font-size:14px}.section{margin:0 0 27px;background:var(--paper);border:1px solid var(--line);padding:28px 30px;border-radius:15px;scroll-margin-top:20px}.section-heading{display:flex;justify-content:space-between;align-items:start;gap:20px;margin-bottom:18px}.index{font-size:12px;color:var(--muted);font-family:ui-monospace,monospace}.sub{font-size:13px;color:var(--muted)}.tabs{display:flex;gap:8px;border-bottom:1px solid var(--line);padding-bottom:15px;margin-bottom:22px;flex-wrap:wrap}.tab{border:1px solid var(--line);padding:9px 17px;border-radius:9px;background:white;font-size:13px;color:var(--muted)}.tab[aria-selected=true]{background:var(--ink);border-color:var(--ink);color:#fff}.method{font-size:13px;color:var(--muted);max-width:900px}.legend{display:flex;gap:20px;font-size:12px;margin:19px 0;flex-wrap:wrap}.legend span{display:flex;align-items:center;gap:7px}.legend i{height:8px;width:16px;border-radius:2px}.cpu{background:var(--cpu)}.fast{background:var(--teal)}.bar-chart{margin:0 0 22px;max-width:860px}.bar-row{margin:16px 0 20px}.bar-title{display:flex;justify-content:space-between;font-size:13px;font-weight:600;margin:0 0 7px}.bar-title span{font:700 12px ui-monospace,monospace;color:var(--teal)}.bar-track{position:relative;height:19px;margin:5px 60px 5px 0;background:linear-gradient(to right,transparent 24.8%,#eff2f2 25%,transparent 25.2%,transparent 49.8%,#eff2f2 50%,transparent 50.2%,transparent 74.8%,#eff2f2 75%,transparent 75.2%)}.bar{height:12px;border-radius:3px;position:relative;top:3px}.bar-track b{position:absolute;left:calc(100% + 11px);top:0;font:500 12px/19px ui-monospace,monospace;color:var(--muted)}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;min-width:510px}th{text-align:left;font-size:12px;color:var(--muted);font-weight:600;padding:11px 12px;background:#f4f7f6}td{padding:13px 12px;border-bottom:1px solid #e9efed}tr:last-child td{border-bottom:0}.num{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap}.accent{color:var(--teal);font-weight:650}.badge{display:inline-block;border-radius:5px;font-size:11px;font-weight:600;padding:3px 8px;white-space:nowrap}.pass{color:#076b58;background:#e4f2eb}.fail{color:#a23d35;background:#f9eae7}.neutral{color:#5b706c;background:#edf1ef}.warn{color:#925c16;background:#fbf0dc}.note{margin-top:16px;font-size:12px;color:var(--muted)}.delta-track{width:120px;height:8px;background:#eaf0ed;border-radius:5px;overflow:hidden}.delta-track i{display:block;height:100%;background:var(--teal);border-radius:5px}.two-column{display:grid;grid-template-columns:1.2fr 1fr;gap:24px}.small-panel{padding:21px;background:#f5f8f7;border-radius:11px}.small-panel p{font-size:13px;color:var(--muted)}.failure{background:#fbf2ed;border:1px solid #f0ded4}.failure h3{color:var(--red)}.facts{display:grid;grid-template-columns:130px 1fr;font-size:13px;gap:8px 13px;margin:14px 0 0}.facts dt{color:var(--muted)}.facts dd{margin:0;overflow-wrap:anywhere}code{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}.steps{padding-left:20px;margin:0}.steps li{margin-bottom:13px;font-size:13px}.steps strong{display:block;margin-bottom:4px}.steps span{color:var(--muted)}.prob-compare{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:20px 0}.prob-compare strong{font-size:26px;display:block;font-family:ui-monospace,monospace}.prob-compare span{font-size:12px;color:var(--muted)}.architecture{display:grid;grid-template-columns:1fr 45px 1fr;align-items:center;gap:10px;margin:18px 0}.arch{border:1px solid var(--line);padding:16px;border-radius:10px;text-align:center;font-size:13px}.arch b{display:block;margin-bottom:5px}.arch span{color:var(--muted);font-size:12px}.arrow{text-align:center;color:var(--teal);font-size:26px}.arch.npu{border-color:#a9d5c6;background:#eaf5ef}.bullet-list{padding-left:18px;color:var(--muted);font-size:13px}.bullet-list li{margin:8px 0}.foot{color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:23px;display:flex;justify-content:space-between;gap:20px}.sources{display:flex;gap:15px;flex-wrap:wrap;margin-top:14px;font-size:12px}.evidence label{display:block;font-size:12px;margin-bottom:8px}.evidence select{width:100%;max-width:100%;border:1px solid var(--line);border-radius:7px;padding:9px;font-size:12px;background:white;color:var(--ink)}.evidence pre{margin:14px 0 0;padding:18px;background:#183139;color:#dfebe8;max-height:400px;overflow:auto;border-radius:9px;font:11px/1.7 ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}details{margin-top:18px}summary{cursor:pointer;font-size:13px;font-weight:600}.checklist{display:grid;grid-template-columns:repeat(2,1fr);gap:9px;margin-top:18px}.check{padding:10px 13px;border:1px solid var(--line);border-radius:8px;font-size:12px}.check b{color:var(--teal);margin-right:8px}.status-caption{font-size:13px;margin:15px 0 0;color:var(--muted)}
@media(max-width:760px){.masthead-inner{padding:15px 20px}.nav{display:none}.page{padding:28px 18px 40px}.hero{grid-template-columns:1fr;gap:16px}h1{font-size:31px}.stamp{padding:17px}.metrics{grid-template-columns:repeat(2,1fr);gap:9px}.metric{padding:16px}.metric-number{font-size:27px}.section{padding:22px 17px}.two-column{grid-template-columns:1fr}.section-heading{gap:8px}h2{font-size:22px}.facts{grid-template-columns:100px 1fr}.architecture{grid-template-columns:1fr;gap:8px}.arrow{transform:rotate(90deg);line-height:1}.checklist{grid-template-columns:1fr}.foot{flex-direction:column;gap:5px}.tab{padding:8px 11px;font-size:12px}}
@media print{body{background:white;font-size:11px}.page{padding:12px}.masthead,.actions,.tabs,.evidence,.foot .sources{display:none}.hero{grid-template-columns:1fr 230px}h1{font-size:29px}.section{padding:18px;border-radius:0;break-inside:avoid}.perf-panel[hidden]{display:block!important}.perf-panel{margin-bottom:25px;break-inside:avoid}.metric{padding:13px}.metric-number{font-size:26px}.metrics{gap:8px}a{color:inherit}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
</style></head><body>
<header class="masthead"><div class="masthead-inner"><div class="brand">EDGE INFERENCE / 实验记录</div><nav class="nav" aria-label="报告导航"><a href="#performance">性能</a><a href="#precision">输出一致性</a><a href="#rkllm">RKLLM 结论</a><a href="#shutdown">服务状态</a><a href="#reproduction">复现</a><a href="#api">调用示例</a></nav></div></header>
<main class="page"><div class="hero"><div><div class="eyebrow">2026.10.13 · RK3576 · LAN SERVER</div><h1>d3-edge × RK3576<br><span>部署与加速测试报告</span></h1><p class="lede">本轮验证了原始 CPU、优化 CPU、RKLLM 整包转换和 RKNN 矩阵卸载四条路径。可复用的实验方案是 <strong>CPU + RKNN MatMul</strong>：保留原来的非因果注意力逻辑，将较大的文本线性层交给 NPU。</p></div>
<aside class="stamp"><div class="status"><span class="dot"></span>@@STATUS@@</div><p>@@STATUS_DETAIL@@</p><p>自动重启策略保留 no · 手动启动</p><p>核验时间<br>@@TIME@@</p><div class="actions"><button class="button" id="print">打印 / PDF</button><button class="button" id="download">导出数据</button></div></aside></div>
<div class="metrics"><article class="metric"><div class="metric-label">真实 HTTP 单问题提速</div><div class="metric-number">2.37×</div><div class="metric-note">6.69 秒 → 2.82 秒</div></article><article class="metric"><div class="metric-label">RKNN 最大概率 / 分数偏差</div><div class="metric-number">0.00283</div><div class="metric-note">验收容差 ≤ 0.01</div></article><article class="metric"><div class="metric-label">NPU 卸载文本线性层</div><div class="metric-number">78</div><div class="metric-note">原始模型 12 层结构保留</div></article><article class="metric"><div class="metric-label">模型内基准通过用例</div><div class="metric-number">5 / 5</div><div class="metric-note">12 个问题 · 分类选择一致</div></article></div>
<div class="callout"><p><strong>结论：</strong>RKNN 混合路径在已测试的短文本与小图片样本上通过一致性校验。RKLLM 1.3.1 的本次适配路径虽然能运行 NPU，仍未复现 d3 的非因果计算，不能作为可用替代。下文性能均为 <strong>2026-10-13 历史实测</strong>；本次重新启动仅验证接口与输出，不重新计算提速倍数。</p></div>
@@FINAL@@
<section class="section" id="performance"><div class="section-heading"><div><h2>速度提升有多少</h2><p class="sub">不同计时口径分开展示；单位为每个请求的秒数，越低越好。</p></div><span class="index">01 / PERFORMANCE</span></div>
<div class="tabs" role="tablist" aria-label="计时口径"><button class="tab" id="tab-http" role="tab" aria-controls="panel-http" aria-selected="true" data-mode="http">实际 HTTP</button><button class="tab" id="tab-model" role="tab" aria-controls="panel-model" aria-selected="false" data-mode="model">模型内成对基准</button><button class="tab" id="tab-history" role="tab" aria-controls="panel-history" aria-selected="false" data-mode="history">早期 CPU 优化</button></div>@@PANELS@@
<p class="note">模型内基准中，三个问题的文本请求约提速 2.47–2.74 倍，图片约提速 1.43 倍。最终 HTTP 复测中，三个问题耗时 6.09 秒、图片耗时 11.89 秒；这两项没有本轮对应的 CPU HTTP 成对复测，故不计算 HTTP 提速倍数。</p>
<details><summary>服务线程对照与冷启动成本</summary><div class="small-panel" style="margin-top:14px"><p>最初 HTTP 单问题中位数为 4.17 秒；固定辅助工作线程为 4.25 秒，没有明显改善。单独对照发现同一模型在主线程更快，最终让模型初始化、NPU 缓存和推理复用主线程，HTTP 收发在后台处理；最终中位数为 2.82 秒。</p><p>三次最终 HTTP 单问题：2.82 / 3.56 / 2.66 秒。极小浮点数归零设置未加入最终服务。NPU 权重打包及缓存建立在模型基准中约需 6.21 秒；完整服务启动还包含模型加载，未单独测得完整启动时长。</p></div></details></section>
<section class="section" id="precision"><div class="section-heading"><div><h2>输出有没有变</h2><p class="sub">一致性基线为本机优化 CPU。统计概率与 score 的绝对差，不含派生 confidence 字段。</p></div><span class="index">02 / PARITY</span></div>
<div class="table-wrap"><table><thead><tr><th>用例</th><th>最大绝对差</th><th>容差预算占用</th><th>结果</th></tr></thead><tbody>@@ACCURACY@@</tbody></table></div><p class="note">容差为 0.01。最大差值 0.002831 约占容差的 28.3%；当该差值来自概率时相当于约 0.283 个百分点。所有样本的分类选择保持一致，HTTP 复测和不同矩阵尺寸的并发请求也通过。尚无完整输入分布的任务准确率评测。</p>
<div class="architecture"><div class="arch"><b>原始 d3 / PyTorch CPU</b><span>非因果 mask · RoPE · GatedDeltaNet<br>视觉编码 · FP32 readout / softmax</span></div><div class="arrow" aria-hidden="true">↔</div><div class="arch npu"><b>RKNN / NPU</b><span>78 个文本 Linear.forward 的矩阵乘<br>FP16 × FP16 → FP32 → 原 BF16 边界</span></div></div><p class="note">保留原始权重包与 12 层结构，不需要将整个模型重新导出为 RKNN 文件。启动时把矩阵权重转为 FP16 并打包到原生布局，缓存复用；注意力逻辑仍在原 d3 Runtime 内执行。</p></section>
<section class="section" id="rkllm"><div class="section-heading"><div><h2>RKLLM：能转换，未通过语义校验</h2><p class="sub">结论限定于官方 SDK / Runtime 1.3.1 的本次 custom / Qwen3.5 混合注意力路径。</p></div><span class="index">03 / COMPATIBILITY</span></div>
<div class="two-column"><div><ol class="steps"><li><strong>本机 x86 Docker 转换成功</strong><span>ARM64 Mac 的 OrbStack 使用 linux/amd64 仿真，Python 3.12、Torch 2.6.0+cpu、Transformers 5.8.0，官方工具包可加载和导出模型。</span></li><li><strong>原始 W8A8 模型初始化失败</strong><span><code>Tensor blk.2.ssm_conv1d.weight not found</code>。Runtime 的周期层布局假设与官方发布的 12 层 layer_types 不一致；本实验没有额外裁剪。</span></li><li><strong>补入 6 个等价空层后，NPU 可运行</strong><span>18 层 FP16 实验模型完成 10 个 token 请求；单问题最后隐藏向量推理为 2.71–3.47 秒，不含模型加载。补层前后 CPU 隐藏向量差为 0。</span></li><li><strong>输出表现仍匹配因果计算</strong><span>已从导出文件读回 <code>causal=false</code>；NPU 与 CPU 因果计算的隐藏向量余弦相似度为 0.9999998，与正确非因果计算为 0.7706。</span></li></ol></div>
<aside class="small-panel failure"><span class="badge fail">输出一致性未通过</span><h3 style="margin-top:16px">同一个编程请求</h3><div class="prob-compare"><div><span>优化 CPU · code 概率</span><strong>99.92%</strong></div><div><span>RKLLM NPU · code 概率</span><strong>56.01%</strong></div></div><dl class="facts"><dt>最大概率差</dt><dd class="num">0.752699</dd><dt>分类选择一致</dt><dd>2 / 4</dd><dt>NPU 负载峰值</dt><dd>26%</dd></dl><p style="margin-top:18px">硬件执行成功不能等同于正确运行 d3。此次差异来自注意力语义，重新量化本身无法修复；该产物仅作为诊断证据保留。</p></aside></div></section>
@@SUPPORT@@<section class="section" id="validation"><div class="section-heading"><div><h2>验证范围与剩余边界</h2><p class="sub">按实际变更范围验证，所有结果可在文末展开原始记录。</p></div><span class="index">04 / VALIDATION</span></div>
<div class="table-wrap"><table><thead><tr><th>验证项</th><th>记录与边界</th><th>结果</th></tr></thead><tbody>
<tr><td>早期 CPU 优化</td><td>线性 / 卷积 FP32 计算、BF16 输出；样本最大概率或分数差 0.005656</td><td><span class="badge pass">通过 ≤0.01</span></td></tr>
<tr><td>初次官方参考输出比较</td><td>早期记录含 confidence 的最大差为 0.005774，未达当时的 0.001 目标</td><td><span class="badge warn">未达初始目标</span></td></tr>
<tr><td>矩阵算子</td><td>16 个矩阵尺寸 / 动态切换检查；大矩阵约快 2.6–11.2 倍，小矩阵略慢</td><td><span class="badge pass">通过</span></td></tr>
<tr><td>完整 RKNN 混合模型</td><td>5 个用例、12 个问题；每后端 / 用例测量 3 次</td><td><span class="badge pass">通过</span></td></tr>
<tr><td>尺寸与回退</td><td>127/128/129/256/257/512/513 行；超限与非 BF16 精确回退 CPU；原生超限拒绝</td><td><span class="badge pass">通过</span></td></tr>
<tr><td>资源释放</td><td>释放原生缓冲区前恢复 CPU forward，恢复后的调用结果一致</td><td><span class="badge pass">通过</span></td></tr>
<tr><td>HTTP 与并发</td><td>连续单问题、三问题、图片、并发尺寸切换；无效请求返回 422</td><td><span class="badge pass">通过</span></td></tr>
<tr><td>原服务保持</td><td>实验期间原 CPU 容器未替换，复测响应完全一致</td><td><span class="badge pass">通过</span></td></tr>
<tr><td>长文本 / 大批次 / 视频</td><td>超过矩阵行数范围会回退 CPU；未做长文本吞吐或视频性能基准</td><td><span class="badge neutral">性能未验证</span></td></tr>
</tbody></table></div>
<div class="two-column" style="margin-top:22px"><div class="small-panel"><h3>运行环境</h3><dl class="facts"><dt>服务器</dt><dd>NanoPi R76S / RK3576 · 192.168.1.164</dd><dt>CPU / 内存</dt><dd>4× Cortex-A53 + 4× Cortex-A72 / 16 GB</dd><dt>板端系统</dt><dd>Armbian / Debian 13 · RKNPU driver 0.9.8</dd><dt>CPU Runtime</dt><dd>Torch 2.12.0+cpu / Transformers 5.17.0</dd><dt>RKNN Runtime</dt><dd>2.5.0 · 官方代码固定提交</dd><dt>MatMul 核心</dt><dd>AUTO；双核合并设置被拒绝，实测 Core0 工作</dd><dt>RKNN 负载</dt><dd>采样峰值 27%，不能据此推算吞吐上限</dd><dt>运行时内存</dt><dd>CPU 约 1.87 GiB / RKNN 服务约 2.85 GiB</dd></dl></div>
<div class="small-panel"><h3>适用范围</h3><ul class="bullet-list"><li>当前 MatMul 支持最多 512 行：行数 = batch × 补齐后的序列长度。超出时回退 CPU，不截断输入。</li><li>只卸载适合的文本线性层；视觉塔、注意力和扫描继续使用 CPU。</li><li>基准测量按顺序执行。两个服务同时有请求时，共享 CPU 核心 4–7，可能相互争用。</li><li>样本只覆盖短文本、三个问题的批处理和一张小图片；可用结论应限定到已验证范围。</li><li>原始 BF16 → 优化 CPU，以及优化 CPU → RKNN，属于两组不同的一致性基线。</li></ul></div></div></section>
<section class="section" id="shutdown"><div class="section-heading"><div><h2>当前状态：RKNN 服务启动核验</h2><p class="sub">按用户本次要求启动已验证的 RKNN 混合服务，保留 CPU 容器作为关闭状态的备用。</p></div><span class="index">05 / SERVICE</span></div>
<div class="table-wrap"><table><thead><tr><th>容器</th><th>端口</th><th>当前状态</th><th>自动重启策略</th></tr></thead><tbody>@@STOPPED@@</tbody></table></div>
@@LIVE_CHECKS@@
@@LIVE_DETAILS@@<p class="status-caption">核验：@@TIME@@。服务状态是该时刻的核验快照；自动重启策略保持 no，主机或容器重启后需要手动启动。</p>
<details><summary>历史停机命令与保留路径</summary><div class="small-panel" style="margin-top:12px"><p><code>docker update --restart=no decision3-edge decision3-edge-matmul</code><br><code>docker stop decision3-edge decision3-edge-matmul</code></p><dl class="facts"><dt>CPU 部署</dt><dd><code>/opt/decision3-edge/</code></dd><dt>RKLLM 实验</dt><dd><code>/opt/decision3-edge-npu/</code></dd><dt>RKNN 实验</dt><dd><code>/opt/decision3-edge-matmul/</code></dd><dt>本机归档</dt><dd><code>/Users/laji/Downloads/decision3-edge-deployment/</code></dd></dl></div></details></section>
@@REPRODUCTION@@
@@API@@
<section class="section evidence" id="evidence"><div class="section-heading"><div><h2>原始数据与复现证据</h2><p class="sub">数据直接嵌入本 HTML；离线打开即可查看和导出。记录中的运行状态可能是测试时的历史快照，当前状态以本次服务启动核验为准。</p></div><span class="index">06 / EVIDENCE</span></div><label for="source-select">选择原始记录</label><select id="source-select">@@OPTIONS@@</select><pre id="source-json" tabindex="0" aria-label="原始 JSON 数据"></pre>
<div class="sources"><a href="https://huggingface.co/vllm-sr/d3-edge" target="_blank" rel="noopener">模型来源 ↗</a><a href="https://github.com/airockchip/rknn-llm/tree/f7390530443bf84f0394255a449d7cbe81e69d1c" target="_blank" rel="noopener">RKLLM 固定提交 ↗</a><a href="https://github.com/airockchip/rknn-toolkit2/tree/2c723177a94fac65cbc036d722907ddbc91980cf/rknpu2/examples/rknn_matmul_api_demo" target="_blank" rel="noopener">RKNN MatMul 固定提交 ↗</a></div>
<details><summary>模型版本与验证命令</summary><div class="small-panel" style="margin-top:12px"><p><code>vllm-sr/d3-edge @ 20c745653400d191ecacd99c4eafc0c5900dc15e</code></p><p><code>bash /opt/decision3-edge-matmul/build-bridge.sh</code> — 编译通过<br><code>bash /opt/decision3-edge-matmul/run-benchmark.sh all</code> — 模型一致性与性能测试通过<br><code>python3 rknn-matmul-experiment/verify_api.py</code> — 停机前 HTTP / 并发 / CPU 保持测试通过<br><code>micro_test.py / boundary_test.py</code> — 算子与边界测试通过<br><code>ast.parse / bash -n</code> — 实验脚本语法检查通过</p><p>以上为历史验证；本次按用户要求重新启动 RKNN 服务，健康检查及推理输出核验另见 service-activation-verification.json。</p></div></details></section>
<footer class="foot"><span>d3-edge / RK3576 · 测试 2026-10-13 / 状态更新 2026-10-17 · 本地实验结果</span><span>自包含 HTML · 无外部脚本、字体或图表依赖</span></footer>
</main><script id="report-data" type="application/json">@@DATA@@</script><script>
const report=JSON.parse(document.getElementById('report-data').textContent);
const tabs=[...document.querySelectorAll('[role=tab]')];
function activate(key){tabs.forEach(tab=>{const active=tab.dataset.mode===key;tab.setAttribute('aria-selected',String(active));tab.tabIndex=active?0:-1;document.getElementById('panel-'+tab.dataset.mode).hidden=!active;});}
tabs.forEach((tab,index)=>{tab.addEventListener('click',()=>activate(tab.dataset.mode));tab.addEventListener('keydown',e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();const next=tabs[(index+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length];activate(next.dataset.mode);next.focus();}});});activate('http');
const select=document.getElementById('source-select');function renderSource(){document.getElementById('source-json').textContent=JSON.stringify(report.sources[select.value],null,2);}select.addEventListener('change',renderSource);renderSource();
document.getElementById('print').addEventListener('click',()=>window.print());
document.getElementById('download').addEventListener('click',()=>{const blob=new Blob([JSON.stringify(report,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='d3-edge-rk3576-test-data.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);});
</script></body></html>'''
for token,value in {'@@REPRODUCTION@@':reproduction_section,'@@LIVE_DETAILS@@':live_details,'@@STATUS@@':esc(status_title),'@@STATUS_DETAIL@@':esc(status_detail),'@@FINAL@@':final_section,'@@API@@':api_section,'@@LIVE_CHECKS@@':('<div class="checklist"><div class="check"><b>✓</b>8900 健康接口 ready</div><div class="check"><b>✓</b>实际推理选项与基线一致</div><div class="check"><b>✓</b>概率 / score 差 ≤ 0.01</div><div class="check"><b>✓</b>CPU 8899 仍关闭</div></div>' if activation else '<p>等待服务启动后的实际核验，尚未宣称接口可用。</p>'),'@@TIME@@':esc(time_local),'@@PANELS@@':panels,'@@ACCURACY@@':accuracy_rows,'@@STOPPED@@':stopped_rows,'@@OPTIONS@@':evidence_options,'@@DATA@@':data,'@@SUPPORT@@':support_section}.items():
    template=template.replace(token,value)
OUT.write_text(template)
assert '@@' not in template
print(f'{OUT} ({OUT.stat().st_size:,} bytes)')
