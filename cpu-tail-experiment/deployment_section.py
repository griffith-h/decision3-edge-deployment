# -*- coding: utf-8 -*-
"""Render current deployment evidence separately from historical experiments."""
import html

def render(d):
 e=lambda x:html.escape(str(x));t=d['http']['timing'];m=lambda x:f'{1000*x:.2f}'
 scope=d['http']['cases'];rows=''.join(f'<tr><td>{e(k)}</td><td>{v["route"]["input_tokens"]}</td><td>{v["route"]["graph_calls"]}</td><td>{m(v["seconds"])}</td></tr>' for k,v in scope.items() if k in ('single','chinese-0','chinese-1','english','image','novel-math','novel-chat'))
 commands='''# RK宿主机，先冻结工作区（已有发布目录时改用新版本名，不覆盖）
python3 /opt/decision3-edge-graph-deploy-tools-20261018/package.py --release /opt/decision3-edge-graph-20261018-v1
bash /opt/decision3-edge-graph-20261018-v1/run.sh decision3-edge-graph-staging 8901 no
# 开发机器：先验收临时端口
python3 cpu-tail-deployment/verify_http.py --port 8901 --count 20 --output staging-http.json
# RK宿主机：切换；失败自动恢复旧服务
python3 /opt/decision3-edge-graph-20261018-v1/cutover.py
# 开发机器：正式100次LAN HTTP和完整回归
python3 cpu-tail-deployment/verify_http.py --port 8900 --count 100 --output production-http.json'''
 rollback='''# RK宿主机：停止新服务，恢复保留的旧三线程容器
bash /opt/decision3-edge-graph-20261018-v1/rollback.sh
curl -fsS http://192.168.1.164:8900/health'''
 curl='''curl -fsS http://192.168.1.164:8900/v1/systemone \\
  -H 'Content-Type: application/json' --data-binary @request.json'''
 return f'''<section id="deployment" class="note"><h2>2026-10-18 实际部署、适用范围与验收</h2>
<p><b>已接管 <a href="http://192.168.1.164:8900/health">192.168.1.164:8900</a></b>，发布 <code>{e(d['release'])}</code>，容器 <code>decision3-edge-graph / {e(d['new_container']['id'][:12])}</code>。旧三线程容器停止保留，可回滚；不再同时监听8900。新服务Docker healthy、restart=unless-stopped。</p>
<div class="cards"><div class="card">正式LAN HTTP100次中位数<div class="big">{m(t['median'])} ms</div>115 token · 单问题 · 并发1</div><div class="card">正式LAN HTTP P95<div class="big">{m(t['p95_nearest_rank'])} ms</div>最小{m(t['min'])} / 最大{m(t['max'])} ms</div></div>
<p>临时8901的20次中位数为{m(d['staging_timing']['median'])}ms。正式100次均保留；其每次图命中12层，证明没有答案缓存。19场景26问题、接口schema/usage、非法输入422、模型列表以及两个不同并发请求通过；与冻结graph-half候选相比，包含confidence的最大绝对差异为 <b>{d['http']['max_delta_vs_frozen']:.8f}</b>。</p>
<h3>适用范围</h3><p>318.51ms是10-17的暖态方法调用实验；上方才是这次实际LAN HTTP结果。MLP图仅支持 <b>batch1、115 token、隐藏维1024</b>。其他长度/形状走CPU融合回退；不是任意请求都318ms，也不保证1秒内。保留全部12层、权重、非因果完整注意力和DeltaNet逻辑；未反编译。</p>
<p>图片和不同长度已通过HTTP回归；视频沿用原接口，本轮未做视频端到端测试。模型计算仍在主线程串行执行；并发请求会排队，不能用单请求结果承诺并发延迟或吞吐。</p>
<div class="scroll"><table><thead><tr><th>正式回归场景</th><th>input_tokens</th><th>图调用数</th><th>该次HTTP ms</th></tr></thead><tbody>{rows}</tbody></table></div><p>表中每个场景只有一次回归计时，包含可能的首次形状准备开销，不是该长度的统计分布；图片耗时也包含媒体处理。</p>
<h3>精度勘误与部署等价性</h3><p>历史0.01门槛实际覆盖概率/score/noul，不含confidence：概率等最大差异0.00753716；confidence补充审计最大0.01002684，choice不变。这里如实分开记录，没有把旧比较器忽略字段的结果写成全字段通过。发布HTTP另外逐值对照冻结候选，包含confidence，结果如上；这不是标注任务准确率测评。</p>
<h3>部署操作</h3><p>固定环境：RK3576 ARM64 Linux、CPU4–7、Torch/OMP3、内存上限8GiB、现有固定镜像和模型快照。升级前先按<a href="../cpu-tail-deployment/README.md">完整部署文档</a>上传工具；新机器先按<a href="../d3-edge-rk3576-install-guide.html">基础安装指南</a>安装，再按<a href="REPRODUCTION.txt">实验复现步骤</a>补齐历史工作区、ARM库和12个MLP图。仅克隆源码不能直接启动；二进制/模型不随Git提交。</p>
<pre style="overflow:auto;background:white;padding:14px"><code>{e(commands)}</code></pre>
<p>发布包先核对39个运行代码/库/图哈希，再冻结到 <code>{e(d['release_directory'])}</code>，所有运行依赖只读挂载，启动前校验整个包SHA256。不重建镜像、不改时钟/驱动。切换前8901保持旧8900；切换时有短暂中断。已有发布目录或同名新容器会拒绝覆盖，重试前检查实际状态和cutover-result.json。</p>
<h3>回滚与调用</h3><pre style="overflow:auto;background:white;padding:14px"><code>{e(rollback)}</code></pre>
<p>已经实际执行“新→旧→新”演练：旧服务响应与此前原三线程响应完全一致，恢复新服务后响应与冻结候选一致。旧容器ID、配置及发布文件哈希保留；只按切换改变启停状态和restart策略。恢复新版本的命令见完整部署文档，避免同时争用8900。</p>
<pre style="overflow:auto;background:white;padding:14px"><code>{e(curl)}</code></pre><p>请求格式与原接口一致，仓库根目录有request.json。健康接口展示release、图累计命中/回退和最近请求计时。实际发布证据：<a href="../cpu-tail-deployment/deployment-result.json">部署汇总</a> · <a href="../cpu-tail-deployment/production-http.json">正式100次和19场景原始数据</a> · <a href="../cpu-tail-deployment/rollback-rehearsal.json">回滚演练</a> · <a href="../cpu-tail-deployment/VERIFY.txt">验证命令与结果</a>。</p></section>'''
