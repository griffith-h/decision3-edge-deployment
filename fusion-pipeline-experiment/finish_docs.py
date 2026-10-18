"""Refresh shared documentation from validated records; never touches the board."""
from pathlib import Path
import json,statistics
P=Path(__file__).resolve().parent;root=P.parent
load=lambda n:json.loads((P/n).read_text());s=load('logs/summary.json');v=load('logs/validation.json');reg=load('logs/final-regression.json');pro=load('logs/profile-selected.json');short=load('logs/short-formal.json')
assert v['passed'] and reg['qualified'] and not s['execution']['full640']
ms=lambda x:f'{x*1000:.2f}'
c=s['candidate'];b=s['baseline'];a=s['current_maxima'];multi=reg['cases']['multi-wide']['vs_current']
rr=[x for x in pro['records'] if x['mode']=='candidate']
comp={k:statistics.median(x['components'].get(k,{}).get('inclusive',0) for x in rr) for k in ('attention_entry','scan_entry','scan_c','mlp','projection_rows','bridge.mm_prepare_mlp')}
without=statistics.median(x['seconds']-x['components']['attention_entry']['inclusive'] for x in rr)
state='目标达成' if s['goal_met'] else '1秒目标未达成'
section=f'''## 第五阶段：验收完成，{state}

- 最优已测精度合格候选为 fusion-pipeline-experiment/selected_owner.py，冻结配置为 execution.json。三轮独立无端口容器、每轮30组交错配对，合并90份候选P50 **{ms(c['median'])}ms**、P95 **{ms(c['p95_nearest_rank'])}ms**；固定validate_mlp P50 {ms(b['median'])}ms，下降{s['reduction']*100:.2f}%。这是B1/T615暖态方法耗时，不是HTTP/并发SLA，低于1秒的目标没有达到。
- [报告第五阶段](mlp-length-experiment/report.html#fusion-pipeline) · [复现](fusion-pipeline-experiment/REPRODUCTION.md) · [核验命令与结果](fusion-pipeline-experiment/VERIFY.txt) · [原始汇总](fusion-pipeline-experiment/logs/summary.json)。板端 /opt/decision3-edge-fusion-pipeline-20261018，独立容器 decision3-fusion-pipeline-test 已退出；线上未替换，本轮未提交、未推送。
- 采用公开动态QK/PV MatMul的FP32输出、自有准备/FP32 softmax/转换融合；大型投影615行一次提交（内部640补零）。原32行块/64列tile扫描仅使用4线程；Torch仍3、CPU4–7。全部真实K/V、非因果mask、权重、12层及六层扫描跨块状态保留。
- 最终615资格恢复FP32投影/MLP输出和原BF16 MLP中间边界，FP16/FP32句柄银行分离。历史R14已使用过FP32输出，本轮用于修复新615内容的双重舍入；不是新硬件能力。其他形状沿用原资格路径，短多batch另保留精度修复。
- 私有D3.logits末层优化：前11层全部615位置，第12层K/V仍615，query/逐行MLP/残差/Norm只算消费者使用的末行。公开encoder仍返回完整hidden，末行MLP使用原CPU及BF16边界。结构/权重版本/方法/钩子/模式改变旁路；错误丢弃私有结果，整次原logits重算并持续禁用。没有截断输入。
- 34例相对单独新算original current全部通过原门槛，包含13份不同内容615输入及12份额外115输入。最大Δ概率{a['probability_max_delta']:.8f}、Δscore{a['score_max_delta']:.8f}、Δnoul{a['noul_max_delta']:.8f}；choice一致。confidence独立报告，最大差{a['confidence_max_delta']:.8f}，不纳入原0.01门槛。multi-wide score由既有约0.01745降为{multi['score_max_delta']:.8f}。数值一致性不等于业务准确率。
- 更快FP16组合与缩减FP32范围在新615内容上超门槛，未采用。新640 MLP图组合消融未胜出；staging、投影拆列、双head合并、融合扫描、NEON预测/状态核、融合卷积等负面证据保留。早期动态B未同步、K624对齐、隐藏allclose和测试脚本失败均有原日志/源码，不把失败包装为硬件极限。
- 独立五组profile：注意力{ms(comp['attention_entry'])}ms，其他投影{ms(comp['projection_rows'])}ms，扫描{ms(comp['scan_entry'])}ms（原生{ms(comp['scan_c'])}ms），MLP{ms(comp['mlp'])}ms（原边界准备{ms(comp['bridge.mm_prepare_mlp'])}ms为嵌套子项）。各项中位数和重叠worker时间不能相加。假设整个注意力为零，其余仍约{ms(without)}ms；仅继续优化注意力不足以到1秒。这不是未来实现的硬件下限。
- 115行另做12组：baseline P50 {ms(short['timing']['baseline']['median'])}ms、候选{ms(short['timing']['candidate']['median'])}ms；全部慢样本保留，不计入615目标。
- 最终故障注入、模式旁路、持续禁用、重复关闭通过。三轮均106个engine句柄和1037个FD暖态稳定，自有MatMul关闭后为0；RSS高水位约6.33GiB，不代表全部驱动内存。板端252项运行资产SHA前后相同；线上身份/启动时间/配置和554个发布文件SHA不变，健康、五次完整HTTP响应一致。未测多Owner并发、任意输入或业务准确率。

'''
path=root/'WORK_STATUS.md';text=path.read_text();start=text.index('## 第五阶段');end=text.index('## 已完成四轮实验',start);text=text[:start]+section+text[end:]
tick=chr(96);text=text.replace('统一报告由 '+tick+'python3 long-tail-experiment/update_report.py'+tick+' 重建','统一报告由 '+tick+'python3 fusion-pipeline-experiment/update_report.py'+tick+' 重建')
text=text.replace('## 当前最优独立候选与边界','## 第四阶段历史参照与旧边界')
text=text.replace('[service-after.json](long-tail-experiment/service-after.json)','[service-after.json](fusion-pipeline-experiment/service-after.json)')
start=text.index('## 继续优化的候选方向');end=text.index('## 复现、检查与资产位置',start)
text=text[:start]+f'''## 剩余瓶颈与继续条件

本轮完成注意力融合/搬运、投影提交/转换及独立工作调度对照。精度合格路径的成本分散在完整扫描、原BF16边界MLP、其他投影及CPU准备/残差。已有小块、staging、拆列和范围缩减未提供同时满足精度与1秒的路线。P50还需削减约{max(0,c['median']-1)*1000:.0f}ms，并同时降低P95。

下一轮需要新的可验证执行机制，例如保留原BF16边界的更深融合或显著减少原生MLP/投影计算与搬运；先证明公开SDK支持、数值及生命周期，再做端到端。不能直接回用已失败的FP16图、放宽0.01门槛或按失败样本内容旁路。当前结论限固定SDK、频率和四CPU资源，不是RK3576硬件理论下限，也不支持承诺200ms。

'''+text[end:]
if 'python3 fusion-pipeline-experiment/validate_records.py\n' not in text:text=text.replace('python3 long-tail-experiment/validate_records.py\n','python3 fusion-pipeline-experiment/validate_records.py\npython3 fusion-pipeline-experiment/check_report.py\npython3 long-tail-experiment/validate_records.py\n')
path.write_text(text)
path=root/'README.md';lines=path.read_text().replace('四轮未部署实验','五轮未部署实验').splitlines()
for i,line in enumerate(lines):
 if line.startswith('[多长度推理 HTML 报告]'):
  lines[i]=f'[多长度推理 HTML 报告](mlp-length-experiment/report.html#fusion-pipeline) 保留五轮实验。最新615 token暖态方法对照：固定validate_mlp P50 {ms(b["median"])}ms → 最优已测精度合格候选 {ms(c["median"])}ms，P95 {ms(c["p95_nearest_rank"])}ms，下降{s["reduction"]*100:.2f}%；{state}。34例原current数值门槛通过，confidence独立报告，multi-wide旧score超差已修复；不代表业务准确率。未部署、未推送，适用范围和原始记录见报告第五阶段。'
path.write_text('\n'.join(lines)+'\n')
path=P/'REPRODUCTION.md';text=path.read_text().replace('# 第五阶段：融合与流水（进行中）',f'# 第五阶段：融合与流水（验收完成，{state}）')
text=text.replace('阶段结论及最终复现命令待正式验收后补全；目前不能据此宣称低于1秒或全部精度通过。',f'最终615行P50 {ms(c["median"])}ms、P95 {ms(c["p95_nearest_rank"])}ms；原34例数值门槛通过，但1秒目标未达到。详见统一报告第五阶段、logs/summary.json及VERIFY.txt。')
suffix='\n历史筛选需使用对应logs/*-sources快照和当时execution配置，在另一个独立目录复现；不要把最终selected_owner替代早期研究版本后直接运行旧筛选脚本。logs/final-sources保存实际最终运行源码；数据采集后的本地文档生成器只改变呈现及复核元数据，不改计时程序和原始输出。\n'
if suffix not in text:text+=suffix
path.write_text(text)
path=P/'VERIFY.txt';text=path.read_text().replace('第五阶段核验记录（最终结果待补）','第五阶段核验记录（最终验收完成，1秒目标未达成）')
replacement=f'''最终验收结果：
- bash build-selected.sh：采用路径构建成功，无编译警告。
- bash run4.sh test_mixed.py：两种输出类型10份矩阵记录、dense/scatter及有限65496近饱和回退通过，六个测试句柄销毁。
- bash run4.sh test_selected.py：正常5层NPU注意力/11层615精度MLP/1次末层readout；模式旁路、整次logits及注意力回退、持续禁用、重复关闭及原方法恢复通过。
- bash run4.sh final_regression.py：34例原current门槛与运行检查通过；五个代表场景的baseline受检数值与冻结记录相等。
- bash run4.sh profile_selected.py：独立五组完整配对，10份完整输出保存；不混入正式分布。
- bash run4.sh formal_selected.py --round N --pairs 30（N=1,2,3）：三轮180份实际新推理，90份候选全部原门槛通过。合并P50 {ms(c['median'])}ms、P95 {ms(c['p95_nearest_rank'])}ms；对照P50 {ms(b['median'])}ms，下降{s['reduction']*100:.2f}%。1秒未达成；没有丢样本，基线max2346.80ms及候选max1797.03ms均保留。
- 三轮暖态engine句柄106→106、FD1037→1037，关闭engine/native attention句柄0；RSS高水位约6.33GiB，不代表全部驱动内存。
- bash run4.sh short_selected.py --pairs 12：115行另12组，P50 {ms(short['timing']['baseline']['median'])}→{ms(short['timing']['candidate']['median'])}ms；不计入615目标。
- python3 fusion-pipeline-experiment/capture_artifacts.py before / after：252项板端运行源码/配置/库/图/SDK/原模型资产一致，哈希时实验进程均已退出；108份可用本地资产与板端逐项匹配。
- python3 fusion-pipeline-experiment/preserve_service.py after：线上身份/启动时间/配置及554个发布文件SHA相同，健康、五次完整HTTP输出一致，无独立实验容器。
- python3 fusion-pipeline-experiment/validate_records.py：重算{v['outputs_checked']}条保存记录差异及正式P50/P95通过，含归档副本；排除字节相同副本后{v['record_comparisons_excluding_identical_archives']}条。两者均不是独立推理次数，副本映射及早期失败保存在validation.json。
- confidence最大差{a['confidence_max_delta']:.8f}独立报告；原门槛只包含choice及概率/score/noul≤0.01，全部34例通过。score差不是概率差，没有业务准确率结论。
- 报告、旧历史区、语法、shell、差异及视觉核验见logs/local-checks.log和report-visual-check.txt。没有部署、提交或推送。'''
text=text.replace('最终验收、命令和结果：待final-regression/formal/profile/service/artifacts/local validation完成后补全。',replacement);path.write_text(text)
print('STATUS README REPRODUCTION VERIFY UPDATED')
