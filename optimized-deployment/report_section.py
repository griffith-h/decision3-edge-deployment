"""Latest deployment section, kept separate from the historical experiment charts."""
import json

def build(root, esc, code_block, sources):
    def read(name):
        result=json.loads((root/name).read_text())
        sources[str(name)]=result
        return result
    result=read('optimized-deployment/deployment-result.json')
    http=read('latency-experiment/logs/deployed-production-http.json')
    tail=read('latency-experiment/logs/tail-reserved.json')
    previous=read('latency-experiment/logs/http-reserved.json')
    read('latency-experiment/result-round2.json')
    read('optimized-deployment/cutover-verification.json')
    read('optimized-deployment/staging-lifecycle.json')
    read('latency-experiment/logs/validation-final.json')
    read('latency-experiment/logs/health-client-cost.json')
    read('latency-experiment/logs/deployment-staging-http.json')
    for name in ('server.py','run.sh','rollback.sh','README.md'):
        path='optimized-deployment/'+name
        sources[path]={'source':(root/path).read_text()}
    for name in ('cpu_attention_optimized.py','native_norm.py','native_norm.c'):
        path='latency-experiment/'+name
        sources[path]={'source':(root/path).read_text()}
    rows=''.join(f'<tr><td>{esc(name)}</td><td>{x["count"]}</td><td class="num">{x["median"]:.3f}</td><td class="num">{x["p95_nearest_rank"]:.3f}</td><td class="num">{x["max"]:.3f}</td><td>{x["below_one_second"]}/{x["count"]}</td></tr>' for name,x in [('部署前独立 8901 · 三线程',previous),('正式 8900 · 三线程（本次）',http)])
    cases=tail['case_checks']['cases']
    case_rows=''.join(f'<tr><td>{esc(k)}</td><td class="num">{v["max_delta_vs_original_rknn"]:.8f}</td><td class="num">{v["max_delta_vs_cpu"]:.8f}</td><td>一致</td></tr>' if 'max_delta_vs_cpu' in v else f'<tr><td>{esc(k)}</td><td class="num">{v["max_delta_vs_original_rknn"]:.8f}</td><td>无 CPU 参考</td><td>一致</td></tr>' for k,v in cases.items())
    setup='''# 新机器先按安装指南建立基线环境，检出本仓库到 /opt/decision3-edge-deployment。
# 在 RK3576 root shell 执行；原服务不被覆盖。
set -euo pipefail
cd /opt/decision3-edge-deployment
release=/opt/decision3-edge-optimized-20261017-v1
test ! -e "$release"
mkdir -p "$release/base/sdk"
cp optimized-deployment/{server.py,run.sh,rollback.sh} "$release/"
cp latency-experiment/{cpu_attention_optimized.py,native_norm.py,native_norm.c} "$release/"
cp /opt/decision3-edge-matmul/{cpu_optimized_server.py,rknn_linear.py,libmatmul_bridge.so,request.json,sdk-manifest.json} "$release/base/"
cp /opt/decision3-edge-matmul/sdk/librknnrt.so "$release/base/sdk/"
docker image inspect decision3-edge:arm64-cpu --format '{{.Id}}' > "$release/image-id.txt"
docker inspect decision3-edge-matmul > "$release/original-container.json"
cd "$release"
cc -Wall -Wextra -O3 -fPIC -shared -fopenmp -ffp-contract=off native_norm.c -o libd3_native_norm.so -lm
sha256sum server.py cpu_attention_optimized.py native_norm.py native_norm.c libd3_native_norm.so base/*.py base/*.so base/sdk/*.so > SHA256SUMS'''
    cutover='''cd /opt/decision3-edge-optimized-20261017-v1
# 换成自己的 IP；如模型缓存位置不同，另设置 D3_MODEL_ROOT。
export D3_BIND_IP=192.168.1.164
bash run.sh decision3-edge-optimized-staging 8901 no
curl -f http://192.168.1.164:8901/health
# 重复检查直到 ready；deployment 应显示 release、3 线程及原生算子计数。
# 将下文调用示例端口改为 8901，验证选项与概率后再切换。
docker stop -t 60 decision3-edge-optimized-staging
docker inspect decision3-edge-optimized-staging --format '{{.State.ExitCode}} {{.State.OOMKilled}}'
# 应为 0 false；日志含 SERVICE_STOPPED_CLEANLY。
docker rm -v decision3-edge-optimized-staging
docker update --restart=no decision3-edge-matmul
docker stop -t 60 decision3-edge-matmul
bash run.sh decision3-edge-optimized 8900 unless-stopped
curl -f http://192.168.1.164:8900/health
# 等待 ready 后再调用推理；失败立即按下一段回退。'''
    rollback='''# 板端执行；脚本会等待原服务 ready（自己的机器需修改脚本内 IP）。
bash /opt/decision3-edge-optimized-20261017-v1/rollback.sh

# 再启用已创建的优化容器：
docker stop -t 60 decision3-edge-matmul
docker start decision3-edge-optimized
curl -f http://192.168.1.164:8900/health'''
    reproduce='''# 在能访问 RK 板端的电脑上，仓库根目录执行；不启动测试容器。
python3 latency-experiment/verify_http.py --base http://192.168.1.164:8900 \\
  --count 100 --output my-production-http.json
# 原始逐次数据：latency-experiment/logs/my-production-http.json
# 可另执行算子/7 用例回归，详见 latency-experiment/README.md；
# 此类计算基准不要与线上推理并发，否则互相争用 CPU/NPU。'''
    scripts=''.join('<details><summary>展开 '+esc(name)+'（HTML 内置源码）</summary>'+code_block((root/'optimized-deployment'/name).read_text())+'</details>' for name in ('run.sh','rollback.sh','server.py'))
    return f'''<section class="section" id="optimized-deployment"><div class="section-heading"><div><h2 id="conclusion">最新结论：三线程优化版已部署到 8900</h2><p class="sub">2026-10-17 正式部署、真实回退与再次启用，以下为最新核验。</p></div><span class="index">LATEST / DEPLOYED</span></div>
<p><strong>正式服务：</strong><a href="http://192.168.1.164:8900/health">192.168.1.164:8900</a>，容器 <code>decision3-edge-optimized</code>。本次部署后 100 次串行 HTTP：中位数 <strong>{http['median']:.3f} 秒</strong>、P95 <strong>{http['p95_nearest_rank']:.3f} 秒</strong>、最慢 <strong>{http['max']:.3f} 秒</strong>，低于 1 秒 <strong>{http['below_one_second']}/100</strong>；没有隐藏最慢样本。原 RKNN 容器和 CPU 备用容器已停止保留。</p>
<div class="table-wrap"><table><thead><tr><th>相同单问题；不同测试阶段</th><th>次数</th><th>中位数 / 秒</th><th>P95 / 秒</th><th>最慢 / 秒</th><th>&lt;1 秒</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="note">计时含局域网、HTTP、准备、推理及输出；模型已加载，各 HTTP 组另预热一次，并发 1，固定 115 token、1 个问题。P95 为 nearest-rank。新进程仅预热一次的独立模型测试另为 0.865 / 0.899 / 0.987 秒（中位数/P95/最慢），100/100&lt;1 秒。不同阶段分别列出，未与历史 CPU 单次计时拼接成严格成对提速倍数。</p>
<h3>优化了什么，为什么保留 3 线程</h3><p>78 个文本线性层继续使用 RKNN FP16 矩阵乘。CPU 侧采用自己编写的 ARM64 RMSNorm / 四抽头因果卷积融合算子、128 扫描分块、仅文本的 FP32 SDPA；保持 BF16 舍入边界和不支持路径的回退。原始 12 层、权重、非因果完整注意力、因果 GatedDeltaNet、视觉塔与输出 readout 均保留。无需厂家执行器源码或逆向。</p><p>4–7 四个 A72 核可用，但 Torch 计算线程和 OpenMP 上限均设为 3。4 线程融合版独立模型测试中位数 0.824 秒、最慢 1.397 秒，98/100&lt;1 秒；3 线程的中位数略高但本轮尾延迟更稳定。原服务 Python 健康探针每次消耗约 0.9–1.0 秒 CPU，曾与慢峰重叠；新容器改为每 30 秒 curl 探针。没有关闭探针、改变系统 governor 或使用请求结果缓存。</p>
<h3>精度差异：概率数值变化，不是任务准确率结论</h3><p>最终三线程计算配置已回归 7 个用例 / 14 个问题，分类选择全部一致。相对原 RKNN 最大概率 / score 绝对差 <strong>0.00363007</strong>；相对历史优化 CPU 最大差 <strong>0.00248951</strong>（仅 5 个用例有 CPU 参考），均 ≤0.01。概率差 0.00363007 等于约 0.363 个百分点，不表示准确率下降 0.363%。confidence 与 legend 不纳入此指标。3 线程相对 4 线程候选最大差 0.00483379，线程改变会影响浮点累计顺序。</p>
<div class="table-wrap"><table><thead><tr><th>最终三线程用例</th><th>相对原 RKNN 最大差</th><th>相对历史 CPU 最大差</th><th>分类选择</th></tr></thead><tbody>{case_rows}</tbody></table></div>
<p class="note">以上 7 用例为部署前选定版本的计算回归，本次部署的三份计算算子源码哈希与之相同；这次正式部署额外逐请求验证单问题的 100 次输出，最大差 {max(x['max_delta_vs_historical_cpu'] for x in http['requests']):.8f}。164 个主要算子用例及回退边界断言已通过。本次新增 staging API / 422、推理中 SIGTERM、真实回退与最终端口的 HTTP 核验，没有将旧回归伪称为重新执行。</p>
<div class="small-panel"><h3>1 秒的适用边界</h3><p>只覆盖上述短文本、单问题、并发 1 的已预热场景。7 用例中的三个问题约 2.67–2.74 秒，小图片约 9.67 秒；长文本、视频与高并发没有 1 秒保证。100 个样本不能证明长期硬时限，200ms 尚未达到。冷启动包括完整哈希、加载和 NPU 缓存建立，不计入请求延迟。</p></div>
<h3 style="margin-top:24px">部署步骤与固定版本</h3><p>新 RK3576 机器先按 <a href="d3-edge-rk3576-install-guide.html">从零安装指南</a> 完成 ARM64 Linux / Docker、模型下载与基线 RKNN 安装。从零指南的镜像名、容器名和模型目录与原测试机不同，先按本节末“完整升级说明”的<strong>目录衔接步骤</strong>导出桥接库、建立模型相对链接并设置 D3_MODEL_ROOT，再执行下列命令；该从零衔接未在空白机器完整重跑。以下只复制源码并编译自己的 CPU 算子，不重新构建镜像、不修改原服务目录。模型固定 <code>20c745653400d191ecacd99c4eafc0c5900dc15e</code>，RKNN SDK 固定 <code>2c723177a94fac65cbc036d722907ddbc91980cf</code>，Torch 2.12.0+cpu、Transformers 5.17.0、RKNN Runtime 2.5.0、驱动 0.9.8。需板端 C 编译器/OpenMP（Debian 的 build-essential）。</p>
{code_block(setup)}
<p>实际部署目录 <code>/opt/decision3-edge-optimized-20261017-v1</code>，来源 <code>optimized-deployment/</code> + <code>latency-experiment/</code>，运行时只读挂载；原 SDK/桥接库复制到版本目录的 base。镜像由 image-id.txt 固定为 <code>{esc(result['image_id'])}</code>。同一目录不能边运行边覆盖；新版本另建目录和容器。</p>
{code_block(cutover)}
<p>新容器：4–7 核、3 线程、8 GiB 内存、/dev/dri、--init，自动重启 <code>unless-stopped</code>。原 RKNN 与 CPU 备用：<code>no</code>，保持停止，避免开机端口冲突。切换需要停止旧服务、加载新服务，存在短暂不可用窗口。curl 健康探针每 30 秒一次，启动宽限 180 秒。</p>
<h3>回退已实际演练</h3><p>已执行“优化版 8900 → 原版 8900 → 优化版 8900”。原容器 ID、Config、HostConfig、镜像和源码均保留；回退后的完整示例响应与切换前<strong>完全一致</strong>。原版最终处于停止状态，优化版已再次 ready。staging 在请求处理中收到 SIGTERM，当前请求完成、NPU 释放，退出码 0 且非 OOM。</p>
{code_block(rollback)}
<p class="note">回退脚本的 IP 需随接收方机器修改。回退后的原容器仍为 no 自动重启；机器重启后需手动 docker start decision3-edge-matmul。保留旧镜像、容器、模型缓存与 /opt/decision3-edge-matmul；不要 prune。重新启用优化版使用 docker start，无需重建容器。</p>
<h3>复现线上测试与查看原始证据</h3>{code_block(reproduce)}
<p>部署记录：<code>optimized-deployment/deployment-result.json</code>、<code>cutover-verification.json</code>；线上逐次计时：<code>latency-experiment/logs/deployed-production-http.json</code>。以上和选定配置的完整 7 用例输出均嵌入文末数据选择器，可离线查看、导出。复现自己的结果请更换输出文件名，保留历史证据。</p>
{scripts}
<details><summary>展开完整升级说明（可独立复制）</summary>{code_block((root/'optimized-deployment/README.md').read_text())}</details>
</section>'''
