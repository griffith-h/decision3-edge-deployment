"""Render only audited aggregate evidence; private inputs never enter the report."""
import html,json
from pathlib import Path
ROOT=Path(__file__).parent;DATA=ROOT/'selected-dot-diagnosis'
def load(name):return json.loads((DATA/name).read_text())
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join('---' for _ in head)+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def html_table(head,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def main():
    pmu=load('pmu-summary.json');control=load('factorial-summary.json');tails=load('tail-summary.json');assembly=load('assembly-summary.json')
    assert pmu['passed'] and control['passed'] and pmu['windows']==528 and control['windows']==512
    assert pmu['minimum_running_fraction']==control['minimum_running_fraction']==1.
    lookup={(tuple(r['shape']),r['suite'],r['padding_bytes'],r['alignment64'],r['mode']):r for r in control['rows']}
    shapes=((512,2048,1024),(615,1024,3584));controls=[];bars=[]
    for shape in shapes:
        for mode in ('old','8','16'):
            a=lookup[shape,'core',0,16,mode];b=lookup[shape,'core',64,16,mode]
            controls.append(['×'.join(map(str,shape)),{'old':'原4路','8':'8路','16':'16路'}[mode],f"{a['median_ms_per_call']:.2f} → {b['median_ms_per_call']:.2f}",f"{b['paired_pad_ratio_same_alignment']:.3f}",f"{a['counts_per_selected']['l1d_cache_refill']:.1f} → {b['counts_per_selected']['l1d_cache_refill']:.1f}",f"{b['pad_faster_rounds']}/8"])
            if shape==shapes[0]:
                for item,label,color in ((a,'原行步幅','#e16a55'),(b,'行步幅+64B','#218b83')):
                    v=item['counts_per_selected']['l1d_cache_refill']
                    bars.append(f'<div class="barrow"><span>{html.escape(controls[-1][1])} · {label}</span><div><i style="width:{v/270*100:.2f}%;background:{color}"></i></div><b>{v:.1f}</b></div>')
    ap=lookup[shapes[0],'core',0,16,'16'];bp=lookup[shapes[0],'core',64,16,'16'];ac=lookup[shapes[0],'cache',0,16,'16'];bc=lookup[shapes[0],'cache',64,16,'16']
    metrics=[]
    for label,key,a,b in [('L1数据缓存回填','l1d_cache_refill',ap,bp),('退休指令','inst_retired',ap,bp),('L2缓存访问','l2d_cache',ac,bc),('L2数据缓存回填','l2d_cache_refill',ac,bc),('L1数据TLB回填','l1d_tlb_refill',ac,bc)]:
        va=a['counts_per_selected'][key];vb=b['counts_per_selected'][key]
        metrics.append([label,f'{va:.2f}',f'{vb:.2f}',f'{(vb/va-1)*100:+.2f}%'])
    alignment=[]
    for align in (0,16):
        for pad in (0,64):
            r=lookup[shapes[0],'core',pad,align,'16'];alignment.append([f'{align}B',f'+{pad}B',f"{r['median_ms_per_call']:.2f}",f"{r['counts_per_selected']['l1d_cache_refill']:.1f}"])
    tail_rows=[]
    for r in tails['rows']:
        widths=r['widths'];tail_rows.append(['×'.join(map(str,r['shape']))]+[f"{widths[str(w)]['extra_lanes_fraction']:.2%}" for w in (4,8,16)]+[f"{widths['16']['K_hot_instruction_ratio_to_old']:.4f}"])
    loop_rows=[[{'old':'原4路','4':'新库4路','8':'8路','16':'16路'}[r['mode']],r['range'],r['instructions'],r['mnemonics']['ldr'],r['mnemonics']['fmla'],len(r['stack_memory_in_K_loop'])] for r in assembly['loops']]
    headers=['真实形状 M×K×N','内核','耗时ms：原→加间隔','8轮配对耗时比','L1回填/点：原→加间隔','改善轮数']
    lead='主要退化来自权重行步幅引起的L1缓存回填放大；16路还增加尾部重复计算。没有发现K热循环寄存器溢出落栈的证据。'
    conclusion='固定同一二进制、选点、权重值与基地址偏移，仅改变诊断副本的行间隔，16路的L1回填从248.9降到69.6次/点，8轮计时全部改善。退休指令几乎不变，L2回填也几乎不变：证据指向L1局部性/多流争用，不能写成DRAM带宽已耗尽。缓存组冲突是最符合证据的解释，尚未逐物理地址证明每个冲突。'
    boundary='1040个PMU窗口全部有效运行100%、线程集合稳定，重放输出BF16逐位相同。r089的80项ARM检查通过，原请求响应与合格版完全相同，对原CPU非confidence最大差9.7483e-6，原门槛未变。仅诊断，未采用新内核、未跑新候选完整34/18题、未部署；原服务已恢复健康。'
    caveat='计时和计数包含Python封装、后端守卫及OpenMP，并非纯汇编循环；新库4路是封装/编译对照。CPU读数均为2.208GHz；内存控制器仍按原策略在0.534/1.320/1.968/2.736GHz自动调频。全部慢值保留，毫秒数与配对比只描述本次组件对照，不是固定内存频率的精确因果占比，也不是整请求收益；尚未证明能省1秒。没有硬件停顿分类计数，不能将剩余周期精确分摊到load-use、发射队列或寄存器重命名压力。'
    next_step='下一步应验证缓存友好的派生权重布局，并减少尾部无用计算；保留原4路作基线，不能直接全量启用16路。先把副本准备、额外内存与生命周期计入成本，证明整请求省1秒以上，再按原门槛跑完整34例和18题。选点、每输出K求和顺序、模型与数值门槛都保持。'
    md=f'''# 选点重算为什么变慢：访存证据，而非继续调宽度

2026-10-20 · r089/r090 · D3决策模型 · RK3576 CPU4–7（Cortex-A72）· Torch3线程

**{lead}**

## 1. 最有区分力的对照

{conclusion}

固定基地址偏移16B，行步幅从K个BF16元素改为K+32；值、选点、K顺序、内核和线程不变。每布局/内核8轮均衡轮换；“配对比”是同轮新旧耗时比的中位数，和两列独立中位数之比不必相等。下表是core事件组，cache事件组独立重复支持同方向。

{table(headers,controls)}

## 2. 拆开对齐与行步幅

r089首个padding对照同时将原权重16B偏移变成0B，存在混杂，已明确保留。r090固定2×2对照补齐；下表为512×2048×1024的16路。单独改基地址对齐不能消除退化，增加行间隔在两种对齐下都有效。

{table(['基地址mod64','行步幅增加','中位耗时ms','L1回填/点'],alignment)}

同样16B偏移，16路内核的其他计数如下。不同事件组分别重复8轮，不能把跨组计数当成同一窗口。

{table(['计数/选点','原行步幅','+64B间隔','变化'],metrics)}

## 3. 寄存器落栈：K热循环没有

冻结r088二进制反汇编，无重新编译。计数单位为一轮K+=4，不同内核分别计算4/8/16个输出。16路在K循环结束后有输出临时数组写栈，不能误认成循环内spill。

{table(['内核','ELF K循环地址','指令条数','load条数','FMLA条数','栈访存'],loop_rows)}

这只能排除“热循环反复落栈”这个具体解释，不能证明不存在其他寄存器/调度约束。

## 4. 尾部重复计算抵消指令收益

源码对每行最后一组仍算满4/8/16路，多余通道重复最后选点。按真实9份选点分布核算：16路额外计算约9.3%～25.2%，原4路约2.1%～5.5%。

例如512×2048×1024：4路额外3.17%，16路额外16.18%；结合实际汇编，预测K热循环指令增加2.62%，PMU退休指令实际增加约2.45%（r089）。因此“每输出静态指令更少”不能代替真实执行量。

{table(['形状 M×K×N','4路额外计算','8路额外计算','16路额外计算','16路K指令/原4路'],tail_rows)}

## 5. 数值与结论边界

{boundary}

{caveat}

## 6. 下一步

{next_step}

[诊断源码、方法与复现](selected-dot-diagnosis/README.md) · [31秒请求耗时账本](REQUEST_COST.html)

硬件解释参考：[Arm Cortex-A72 TRM](https://documentation-service.arm.com/static/60368ce38f952d2e4134dc2e)（L1D32KiB、2路、64B缓存线）；计数接口参考：[perf_event_open](https://www.man7.org/linux/man-pages/man2/perf_event_open.2.html)。本轮计数来自实际A72 PMU，主因判断基于上述实测对照，未从手册推导速度。
'''
    if (ROOT/'SELECTED_LAYOUT_TRIAL.html').exists():
        md=md.replace('## 1. 最有区分力的对照','**后续r091/r092结果：** 已实现布局缓存和尾部分派，但计入准备成本后未过1秒组件潜力门槛，已停止，未采用新布局。见[实验结果](SELECTED_LAYOUT_TRIAL.html)。\n\n## 1. 最有区分力的对照')
    (ROOT/'SELECTED_DOT_DIAGNOSIS.md').write_text(md)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>选点重算变慢的原因</title><style>body{margin:0;background:#f5f7fb;color:#172033;font:16px/1.65 system-ui,sans-serif}main{max-width:1080px;margin:auto;padding:26px}h1{font-size:32px;line-height:1.25}h2{font-size:23px}section,.card{padding:22px;border-radius:14px;background:#fff;margin:18px 0}.cards{display:flex;gap:14px;flex-wrap:wrap}.card{flex:1;min-width:180px;margin:0}.card strong{display:block;font-size:26px}.note{font-size:14px;color:#586579}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:right;border-bottom:1px solid #e6eaf0;white-space:nowrap}td:first-child,th:first-child{text-align:left}.barrow{display:grid;grid-template-columns:155px 1fr 50px;gap:10px;align-items:center;margin:12px 0;font-size:14px}.barrow div{height:22px;background:#f0f3f8;border-radius:4px;overflow:hidden}.barrow i{height:100%;display:block}.barrow b{text-align:right}a{color:#2264c9}summary{cursor:pointer;font-weight:600}code{font-size:13px}@media(max-width:600px){main{padding:16px}h1{font-size:27px}.barrow{grid-template-columns:125px 1fr 38px;font-size:12px}section{padding:16px}}</style><main>'''
    page+='<p class="note">r089/r090 · RK3576 Cortex-A72 · 固定二进制 · 数值门槛不变</p><h1>为什么越并行，反而越慢？</h1><p>'+html.escape(lead)+'</p>'
    page+='<div class="cards"><div class="card">主要证据<strong>L1缓存回填放大</strong></div><div class="card">K热循环栈访存<strong>0处</strong></div><div class="card">有效计数窗口<strong>1040 / 1040</strong></div></div>'
    page+='<section><h2>同一16字节偏移，只改行间隔</h2><p>512×2048→1024 · L1回填次数/选点 · 两种布局使用同一内核</p>'+''.join(bars)+'<p class="note">橙色：原行步幅；绿色：每权重行增加64B间隔。16路回填约减少72%，指令数基本不变。</p><p>'+html.escape(conclusion)+'</p></section>'
    page+='<section><h2>计时也随布局改善</h2>'+html_table(headers,controls)+'<p class="note">同轮配对比取中位数。仅为组件CPU重放；+64B是诊断控制，尚未进入引擎。</p></section>'
    page+='<section><h2>把两个因素拆开</h2>'+html_table(['基地址mod64','行步幅增加','16路耗时ms','L1回填/点'],alignment)+'<p class="note">r089首个对照存在对齐混杂，已保留并补做r090固定2×2对照。只改基地址对齐未消除退化。</p><details><summary>查看L2、指令与TLB证据</summary>'+html_table(['计数/选点','原步幅','+64B间隔','变化'],metrics)+'</details></section>'
    page+='<section><h2>寄存器没有在K循环反复落栈</h2>'+html_table(['内核','K循环地址','指令','load','FMLA','栈访存'],loop_rows)+'<p class="note">同一冻结ELF，未重新编译。16路循环外的输出暂存写栈不属于K循环spill；其他调度/寄存器约束尚未精确量化。</p></section>'
    page+='<section><h2>另一个代价：尾部也算满16路</h2><p>真实选点下，16路多算约<strong>9.3%～25.2%</strong>；原4路约2.1%～5.5%。</p><p class="note">K=2048这一形状：预测热循环指令增加2.62%，PMU实测退休指令增加约2.45%。</p><details><summary>展开9种形状的尾部浪费</summary>'+html_table(['形状','4路多算','8路多算','16路多算','16路K指令/原'],tail_rows)+'</details></section>'
    page+='<section><h2>哪些结论可以落地？</h2><p>'+html.escape(next_step)+'</p><details><summary>数值检查与测量边界</summary><p>'+html.escape(boundary)+'</p><p>'+html.escape(caveat)+'</p><p>缓存组冲突/多流争用是解释，尚未逐物理地址证明；未证明DRAM带宽饱和。</p></details><p><a href="SELECTED_DOT_DIAGNOSIS.md">完整Markdown</a> · <a href="selected-dot-diagnosis/README.md">方法与复现</a> · <a href="REQUEST_COST.html">返回31秒账本</a></p></section></main></html>'
    if (ROOT/'SELECTED_LAYOUT_TRIAL.html').exists():
        page=page.replace('<section><h2>同一16字节偏移，只改行间隔</h2>','<section><h2>后续实验已完成：未过继续门槛</h2><p>r091/r092数值核验通过，但计入准备后，原4路布局方案的整请求条件估算约0.45秒，组合方案约0.09秒。没有候选整请求实测，已按预设门槛停止。</p><a href="SELECTED_LAYOUT_TRIAL.html">查看缓存成本、尾部分派与原4路补充对照</a></section><section><h2>同一16字节偏移，只改行间隔</h2>')
    (ROOT/'SELECTED_DOT_DIAGNOSIS.html').write_text(page)
    print('Rendered SELECTED_DOT_DIAGNOSIS.md/html from audited aggregate evidence')
if __name__=='__main__':main()
