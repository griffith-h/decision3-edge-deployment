"""Publish audited aggregates, keeping component estimates separate from model tests."""
import html,json
from pathlib import Path
ROOT=Path(__file__).parent;DATA=ROOT/'rejected-selected-layout-tail'
def load(name):return json.loads((DATA/name).read_text())
def tab(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join('---' for _ in head)+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def htab(head,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'
def main():
    followup_path=ROOT/'selected-layout-request-pairs/summary-r093.json'
    followup=json.loads(followup_path.read_text()) if followup_path.exists() else None
    first=load('summary-r091.json');control=load('summary-r092.json');assert first['passed_numerical'] and control['passed_numerical']
    assert not first['primary_eligible_for_request_pairs'] and not control['primary_eligible_for_request_pairs']
    names={'layout_only':'新4路＋布局副本','tail_only':'16/8/4内核＋原布局','combined':'16/8/4内核＋布局副本','original4_layout':'原4路＋布局副本（补充对照）'}
    results=[*control['modes'],*first['modes']];rows=[];bars=[]
    for r in results:
        ratio=r['weighted_component_ratio'];saving=r['estimated_net_request_saving_seconds'];rows.append([names[r['mode']],f'{ratio:.6f}',f'{(1-ratio)*100:+.2f}%',f"{r['observed_prepare_release_seconds']:.3f}s",f'{saving:+.3f}s','未过'])
        bars.append('<div class="bar"><span>'+html.escape(names[r['mode']])+'</span><div><i style="width:'+str(min(100,ratio/1.5*100))+'%;background:'+('#d16556' if ratio>1 else '#248b81')+'"></i></div><b>'+f'{ratio:.3f}×'+'</b></div>')
    original=control['modes'][0];counts=original['cache_counts'];peak=counts['peak_bytes']/2**20;allocated=counts['allocated_bytes']/2**20
    head=['实现','组件新/旧耗时比','组件耗时下降（正值为减少）','已测准备＋释放','整请求净收益条件估算','>1秒潜力']
    text='数值检查通过，本版没有达到继续条件。缓存友好的行步幅确有局部收益，但计入请求内准备成本后，现有实现缺乏整请求省1秒的证据；组合内核未优于布局对照。已停止组件筛选，没有执行候选整请求配对、完整34例或18题，没有部署。'
    method='保持原模型、选点、2.5e-7带、10%/65,536预算和每输出K连续FMA顺序。权重副本每行增加64B间隔，64MiB请求内LRU上限，先逐出再分配，请求结束/异常时清空；原权重不改。16/8/4主循环仅算完整16、完整8，再交原4路形式处理尾部，不足4最多补3点。'
    limits='这些都是9个代表性真实选点集合的3组AB/BA组件结果，按372次实际选点量加权，再用r087约5.6798秒重算成本等比例估算，减去本轮已测准备/释放。没有候选整请求延迟实测；未计入的初始化与整模型调度交互仍可能改变结果，不能把估算写成已取得的提速。'
    numeric='r091共79项、r092共80项ARM检查通过。两轮各372次/8,063,030个真实选点的私有BF16与FP32位模式检查通过，所有选点指纹与r088相同。返回的是原路径响应，与合格r072完全相同，对原CPU非confidence最大差9.7483e-6，choice/noul/usage/model一致；不等于已完成候选整模型或全样本准确率验收。'
    confound='r091“新4路＋布局副本”使用新编译helper，不能作为纯布局对照；“16/8/4＋原布局”也同时改变宽主循环，不能据其退化断言尾部分派本身慢。r092保留相同缓存、步幅、线程及门槛，补测原合格4路ELF＋布局副本，单独分离布局效果。旧证据完整保留，没有调宽度、padding或缓存容量找分数。'
    memory=f'原4路补充对照：{control["unique_weights"]}份权重，原BF16数据合计{control["source_bf16_bytes"]/2**20:.1f}MiB；派生布局累计分配{allocated:.3f}MiB，同时存活峰值{peak:.3f}MiB。{counts["builds"]}次构建、{counts["hits"]}次命中（含27次组件计时调用）、{counts["evictions"]}次逐出，没有重复构建同一模块，没有超预算旁路。累计分配量不是峰值内存，也不能把逻辑缓存清空当RSS立即归零。'
    md=f'''# 缓存布局与尾部处理：数值通过，收益未达继续门槛

2026-10-20 · r091/r092 · RK3576 · 仅实验，未部署

**{text}**

## 改了什么

{method}

## 组件结果与条件估算

{tab(head,rows)}

{limits}

## 对照纠正

{confound}

## 准备与内存成本

{memory}

准备/复制/逐出及最终释放已测并从估算扣除，缓存命中守卫和分派在组件计时内。每个完整请求都应从空缓存开始，准备/逐出/释放纳入完整墙钟，另扣一次性安装成本；本轮组件门槛未过，因此没有执行这一步，更没有启动34/18题。

内存结果来自组件诊断（r091同时维护两套带布局缓存，r092仅一套），不能外推为候选完整18题内存。cgroup memory.peak是全轮高水位，包含初始化、预热、隔离检查及模型；与RSS有不同共享内存口径，不相加，也不将前后高水位差当候选独占增量。原始数据在私有日志。

## 数值与运行状态

{numeric}

两轮父模型上下文True/Torch3/highest/autocastFalse、句柄关闭0、OOMFalse、控制器退出0、原同一服务容器恢复healthy。原0.01/choice/noul及保护1e-6不降低。活动probe已恢复，候选源码仅归档；完整1,000条和定时任务保持暂停。

## 这次能支持什么结论

- +64B布局可改善已测组件，但这份实现没有足够收益进入下一阶段。
- 消除16路尾部重复计算不保证整体更快；较宽主循环、代码生成及缓存行为仍有成本。本轮未新增PMU分解，不能编造新的微架构根因。
- 72份权重各只复制一次，调大缓存不会减少这条请求的复制量；不继续盲调容量。
- 该结果不是该方向或RK3576的理论上限。后续方案须有新机制证据，不拿组件估算当整请求收益。

[实际源码、复现与命令](rejected-selected-layout-tail/README.md) · [上一轮瓶颈诊断](SELECTED_DOT_DIAGNOSIS.html)
'''
    (ROOT/'SELECTED_LAYOUT_TRIAL.md').write_text(md)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>缓存布局与尾部实验结果</title><style>body{margin:0;background:#f5f7fb;color:#182439;font:16px/1.7 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:28px}h1{font-size:30px;line-height:1.3}h2{font-size:23px}.cards{display:flex;gap:14px;flex-wrap:wrap}.card,section{padding:22px;background:white;border-radius:14px;margin:18px 0}.card{flex:1;min-width:180px;margin:0}.card strong{display:block;font-size:25px}.note{font-size:14px;color:#59677a}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}td,th{padding:8px;text-align:right;white-space:nowrap;border-bottom:1px solid #e5eaf0}td:first-child,th:first-child{text-align:left}.bar{display:grid;grid-template-columns:250px 1fr 65px;gap:12px;align-items:center;margin:14px 0;font-size:14px}.bar div{height:22px;background:#edf1f7;border-radius:4px;overflow:hidden}.bar i{display:block;height:100%}.bar b{text-align:right}a{color:#2465c8}summary{cursor:pointer;font-weight:600}@media(max-width:650px){main{padding:16px}.bar{grid-template-columns:150px 1fr 55px;font-size:12px}section{padding:16px}}</style><main>'''
    page+='<p class="note">r091/r092 · 2026-10-20 · 同一模型/选点/K顺序/数值门槛</p><h1>数值过了，收益还没过门槛</h1><p>'+html.escape(text)+'</p>'
    if followup:
        note=f'r093已补跑3组完整请求，净节省中位{followup["median_net_saved_seconds"]:+.3f}秒。下方保留r091/r092组件阶段历史结果。'
        md=md.replace('2026-10-20 · r091/r092 · RK3576 · 仅实验，未部署','2026-10-20 · r091/r092 · RK3576 · 仅实验，未部署\n\n> '+note+' [查看完整请求实测](SELECTED_LAYOUT_REQUEST_PAIRS.html)',1)
        (ROOT/'SELECTED_LAYOUT_TRIAL.md').write_text(md)
        page=page.replace('<main>','<main><section><b>'+html.escape(note)+'</b><p><a href="SELECTED_LAYOUT_REQUEST_PAIRS.html">查看3组完整请求实测、准备与内存成本 →</a></p></section>',1)
    page+=f'<div class="cards"><div class="card">继续条件<strong>整请求省 &gt;1秒</strong></div><div class="card">本次执行到<strong>组件筛选</strong></div><div class="card">单套缓存活跃峰值<strong>{peak:.1f}MiB</strong></div></div>'
    page+='<section><h2>改善布局有效，加上尾部没有额外收益</h2><p class="note">横条为加权组件新/旧耗时比，越短越好；1.0×为原路径。不是整请求延迟。</p>'+''.join(bars)+htab(head,rows)+'<p class="note">'+html.escape(limits)+'</p></section>'
    page+='<section><h2>副本不是免费的</h2><p>'+html.escape(memory)+'</p><p>请求内清空、首次复制、逐出和最终释放都纳入成本。组件潜力门槛未过，不通过拉长缓存寿命隐藏准备开销。</p></section>'
    page+='<section><h2>为什么补了原4路对照？</h2><p>'+html.escape(confound)+'</p></section>'
    page+='<section><h2>数值与运行状态</h2><p>'+html.escape(numeric)+'</p><p>两个控制器正常退出，原服务健康，缓存及句柄清理通过。完整34例、18题和部署均未启动。</p><details><summary>查看实现与证据边界</summary><p>'+html.escape(method)+'</p><p>候选内核与私有数据布局发生变化，原数值规则没有变化。未新增PMU，不能把本轮退化归因到未测的调度或寄存器因素；内存高水位包含初始化与预热，不能当单个候选增量。</p></details><p><a href="SELECTED_LAYOUT_TRIAL.md">完整Markdown</a> · <a href="rejected-selected-layout-tail/README.md">源码与复现</a> · <a href="SELECTED_DOT_DIAGNOSIS.html">返回瓶颈诊断</a></p></section></main></html>'
    (ROOT/'SELECTED_LAYOUT_TRIAL.html').write_text(page);print('Rendered SELECTED_LAYOUT_TRIAL.md/html')
if __name__=='__main__':main()
