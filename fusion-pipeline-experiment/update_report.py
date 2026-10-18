# -*- coding: utf-8 -*-
"""Rebuild all historical sections, then append the frozen fifth-stage evidence."""
from pathlib import Path
import json,html,statistics,subprocess,sys
from validate_records import stats
P=Path(__file__).resolve().parent;root=P.parent
load=lambda x:json.loads((P/x).read_text());esc=lambda x:html.escape(str(x));ms=lambda x:f'{x*1000:.2f}'
v=load('logs/validation.json');f=[load(f'logs/formal-{n}.json') for n in (1,2,3)]
reg=load('logs/final-regression.json');pro=load('logs/profile-selected.json');service=load('service-after.json');execution=load('execution.json')
assert v['passed'] and reg['qualified'] and pro['completed'] and service['passed'] and all(x['completed'] for x in f)
b=v['formal_baseline_timing'];c=v['formal_candidate_timing'];goal=c['median']<1 and c['p95_nearest_rank']<1
def table(headers,rows):
 return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def link(name):return f'<a href="../fusion-pipeline-experiment/{name}">{esc(name)}</a>'
formalrows=[]
for j in f:
 bb=j['timing']['baseline'];cc=j['timing']['candidate']
 formalrows.append([j['round'],j['pairs'],ms(bb['median']),ms(bb['p95_nearest_rank']),ms(cc['median']),ms(cc['p95_nearest_rank']),str(j['handles_before'])+'→'+str(j['handles_after'])+'→'+str(j['engine_handles_after_close']),str(j['fds_before'])+'→'+str(j['fds_after']),link(f'logs/formal-{j["round"]}.json')])
formalrows.append(['合并',c['count'],ms(b['median']),ms(b['p95_nearest_rank']),ms(c['median']),ms(c['p95_nearest_rank']),'各轮独立关闭','全部样本','非并发SLA'])
accuracy=[]
for key,row in reg['cases'].items():
 d=row['vs_current'];accuracy.append([esc(key),row['output']['usage']['input_tokens'],*[f'{d[k]:.8f}' for k in ('probability_max_delta','score_max_delta','noul_max_delta','confidence_max_delta')],('一致' if d['choices_unchanged'] else '变化') if any('choice' in a for a in row['output']['answers'].values()) else '无此字段','通过' if d['passed'] else '失败'])
profile=[]
for tag,label in [('attention_entry','完整注意力（含末层单query，若启用）'),('projection_rows','大型投影入口'),('scan_entry','原扫描入口'),('scan_c','原生扫描，嵌套子项'),('mlp','MLP入口'),('conv','因果卷积'),('norm_modules','Norm模块'),('small_linear','其他Linear')]:
 profile.append([label,*[ms(statistics.median(x['components'].get(tag,{}).get('inclusive',0) for x in pro['records'] if x['mode']==mode)) for mode in ('baseline','candidate')]])
profile.append(['未包装余项',*[ms(statistics.median(x['unwrapped'] for x in pro['records'] if x['mode']==mode)) for mode in ('baseline','candidate')]])
profile.append(['带插桩总计',*[ms(statistics.median(x['seconds'] for x in pro['records'] if x['mode']==mode)) for mode in ('baseline','candidate')]])
pilots=[]
for name in ('screen-npu','screen-staging','screen-split','screen-combined','screen-extended','screen-grouped','screen-scan-options','screen-readout'):
 j=load('logs/'+name+'.json');groups={}
 for r in j['records']:groups.setdefault(str(r.get('mode', {'width':r.get('width'),'stage':r.get('stage'),'enabled':r.get('enabled')})),[]).append(r)
 for mode,rr in groups.items():
  pilots.append([esc(name+'/'+str(mode)),len(rr),ms(statistics.median(r['seconds'] for r in rr)),'通过' if all(r['vs_current']['passed'] for r in rr) else '未通过',link('logs/'+name+'.json')])
domains=[]
proof=load('logs/screen-domains.json');ablation=load('logs/ablate-domains.json')
for mode in ('float','float_readout','mlponly_readout','projection_readout'):
 rr=[x for x in ablation['records'] if x['mode']==mode];source=load('logs/screen-float.json') if mode=='float' else proof
 checks=[x for row in source['cases'].values() for x in row['records'] if x['mode']==mode]
 domains.append([esc(mode),len(rr),ms(statistics.median(x['seconds'] for x in rr)),len(checks),f'{max(x["vs_current"]["max_delta"] for x in checks):.8f}', '通过' if all(x['vs_current']['passed'] for x in checks+rr) else '失败'])
candidate_profile=[x for x in pro['records'] if x['mode']=='candidate']
without_attention=statistics.median(x['seconds']-x['components']['attention_entry']['inclusive'] for x in candidate_profile)
without_attention_scan=statistics.median(x['seconds']-x['components']['attention_entry']['inclusive']-x['components']['scan_entry']['inclusive'] for x in candidate_profile)
maxima={k:max(x['vs_current'][k] for x in reg['cases'].values()) for k in ('probability_max_delta','score_max_delta','noul_max_delta','confidence_max_delta')}
section=f'''<section id="fusion-pipeline" class="box"><h2>第五阶段：向615 token低于1秒推进</h2>
<p><b>{'目标达标' if goal else '未达到1秒目标'}。</b>三轮独立容器、每轮30组暖机后交错对照，合并90份候选P50 {ms(c['median'])}ms、P95 {ms(c['p95_nearest_rank'])}ms；固定validate_mlp为{ms(b['median'])}ms，P50下降{(1-c['median']/b['median'])*100:.2f}%。这是batch1、615 token暖态方法耗时，非HTTP/并发SLA。没有部署、提交或推送。</p>
<p>最终候选是当前已测且通过完整回归者，不是全局最优证明。全部重新推理，不缓存答案、不丢慢样本；正式计时无插桩，哈希和服务HTTP核验在实验进程退出后单独运行。</p>
<h3>采用范围与语义</h3>
<p>动态激活MatMul提供FP32输出，CPU FP32 softmax与自有准备/转换融合，全部真实K/V及原mask保留。615行投影扩展原shape银行至640，一次提交替代512+103。四线程只用于原32行块/64列tile扫描；Torch仍三线程、CPU4–7，跨块状态不变。MLP精度修复和末层路径的实际开关以下方冻结配置为准。</p>
<p>若私有readout启用：原D3.logits只取encoder末位置，前11层仍处理全部615位置，第12层保留全部615个K/V和完整非因果mask，仅计算消费者所需query、逐行MLP/残差/Norm；12层权重及六层扫描不变。公开encoder仍返回完整hidden。结构、方法、权重版本、钩子或输入资格变化会旁路；故障丢弃私有结果、重算完整原logits并持续禁用。没有截断输入。</p>
<p>最终组合消融排除了新640行MLP图。额外staging、输出列拆分、共享K/V双head合并、自有NEON预测/状态核、Q/K融合扫描GEMM、卷积候选和FP16概率残差补偿均保留筛选记录；组件快不等于端到端值得采用。</p>
<p>历史R14已使用过公开FP32 MatMul输出；当时115行改成FP16更快并通过七例。本轮恢复已知FP32输出以检查615行双重舍入，用独立句柄银行固定输出类型，保留原非有限/饱和回退。它不是新的硬件能力，也不能将115行旧精度结果推广到615行。</p><details><summary>冻结配置（原始JSON，计时值为秒）</summary><pre>{esc(json.dumps(execution,ensure_ascii=False,indent=2))}</pre></details>
<h3>正式计时及生命周期</h3>{table(['轮','每模式n','基线P50 ms','基线P95 ms','候选P50 ms','候选P95 ms','engine句柄 暖/末/关闭','FD暖/末','原始记录'],formalrows)}
<p>每模式三次暖机、交替先后顺序，两个模式持有相同资源银行。P95采用nearest-rank；句柄和FD暖态稳定，自有MatMul关闭后为0。RSS高水位见原始JSON，不代表全部驱动/NPU内存；未验证多Owner并发。</p>
<h3>精度：以原始current为准</h3>
<p>{len(reg['cases'])}个场景全部新推理并对单独新算current比较，原current在原3线程环境生成并核对已有九例冻结输出。原门槛始终为choice一致、概率/score/noul绝对差≤0.01；confidence独立报告。score范围0–2，不能称为概率差。某字段为0可能是本请求没有询问该类型。数值一致性不等于业务准确率。</p>
{table(['请求','usage tokens','Δ概率','Δscore','Δnoul','Δconfidence','choice','原门槛'],accuracy)}
<p><b>失败保留：</b>扩展12份不同内容的615行请求后，matched615-1/9的旧baseline概率差约0.01574/0.02152，初始快速组合约0.02018/0.01770，均超门槛；原multi-wide的score约0.01745亦失败。最终候选须重新通过，不能只验旧alpha。首版动态B缺同步产生不稳定误差，公开sync/rebind修复；PV的K624对齐错误与初始隐藏allclose失败亦归档。隐藏检查是组件诊断，不替代上述原输出门槛。</p>
<h3>独立profile及条件性瓶颈结论</h3>
<p>每模式5次，独立于正式计时。inclusive入口含子项，原生scan已包含在scan入口，不能重复相加；不同项中位数不要求相加等于总中位数。worker区间会重叠；SDK区间包括提交、等待和转换，并非纯NPU设备时间。readout的profile包装显式登记预期forward，仅用于插桩，不改变正式资格保护。</p>
{table(['组件','固定基线ms','最终候选ms'],profile)}
<p>剩余成本分布在完整扫描、MLP舍入/投影、其他投影、CPU准备及残差等工作。当前分解中假设整个注意力为零，其余仍约{ms(without_attention)}ms；再把整个扫描入口也假设为零，其余约{ms(without_attention_scan)}ms。这是当前执行路径的记账，不是未来融合/并行实现的硬件下限。要进入1秒，P50还需削减约{max(0,c['median']-1)*1000:.0f}ms并同时降低P95。已测列拆分、staging及更多小块融合没有支持足够收益；200ms需要更大幅度、通过精度验收的执行路径变化。结论仅限本轮固定SDK/频率/四CPU及候选实现，不能作为硬件理论下限或未来收益承诺。</p>
<h3>FP32范围与精度修复消融</h3><p>每模式12次暖态轮转计时，正确性另外覆盖13份615内容。float=其他投影与原边界MLP全部FP32输出；mlponly仅MLP，projection仅其他投影且615 MLP仍用原图；readout为末层消费者优化。选择须过原current门槛，并有至少2%收益才增加范围限制。不能采用未过门槛的更快行。</p>{table(['模式','计时n','P50 ms','615内容n','最大概率/score/noul差','当前门槛'],domains)}<p>{link('logs/screen-domains.json')} · {link('logs/ablate-domains.json')}</p><h3>全部小样本筛选</h3><p>不同运行不能跨组直接相减或取最快样本作为正式结果。MLP和精度消融另见下方JSON。</p>
{table(['运行/模式','n','P50 ms','当轮current门槛','原始JSON'],pilots)}
<p>{' · '.join(link(x) for x in ('logs/ablate-mlp.json','logs/screen-precision615.json','logs/regression-research-stack.json','logs/equivalence-before-precision.json','logs/readout-unit-initial.json','logs/probe-npu-attention-unsynced.json'))}</p>
<h3>服务保留与复现</h3>
<p>线上容器{esc(service['id'][:12])}的启动时间、配置及{len(service['files'])}个发布文件SHA不变，健康、五次完整HTTP响应与实验前一致，独立实验容器已退出。所有worker结束后才回退或释放；故障注入、重复关闭、输入旁路及长期禁用见单测。未改变模型权重/层数/非因果mask/扫描状态，未反编译厂商库。</p>
<p>{' · '.join(link(x) for x in ('REPRODUCTION.md','VERIFY.txt','DESIGN.md','logs/final-regression.json','logs/profile-selected.json','logs/validation.json','logs/artifacts-before.json','logs/artifacts-after.json','service-before.json','service-after.json'))}</p>
</section><div class="box note"><b>以下保留前四阶段历史，最新结论以第五阶段为准。</b></div>'''
summary={'goal_met':goal,'baseline':b,'candidate':c,'reduction':1-c['median']/b['median'],'current_maxima':maxima,'regression_cases':len(reg['cases']),'execution':execution,'production_unchanged':True}
(P/'logs/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
subprocess.run([sys.executable,str(root/'long-tail-experiment/update_report.py')],check=True,stdout=subprocess.DEVNULL)
path=root/'mlp-length-experiment/report.html';text=path.read_text().replace('<nav>','<nav><a href="#fusion-pipeline">第五阶段融合与流水</a>',1).replace('</nav>','</nav>'+section,1);text=text.replace('本轮长序列内核','第三阶段长序列内核').replace('本轮投影分块','第二阶段投影分块');path.write_text(text);print('UPDATED',path)
