"""Render audited full-request timings; never substitute component estimates."""
import html,json,statistics
from pathlib import Path

ROOT=Path(__file__).parent
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join('---' for _ in head)+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def htable(head,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(c))+'</th>' for c in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def main():
    data=json.loads((ROOT/'selected-layout-request-pairs/summary-r093.json').read_text())
    assert data['passed_numerical_and_runtime'] and len(data['pairs'])==3
    pairs=data['pairs'];net=data['median_net_saved_seconds'];passed=data['speed_gate_passed']
    assert passed==(net>1.0)
    old=data['median_seconds']['old'];new=data['median_seconds']['new'];install=data['install_seconds']
    timed=[r for r in data['warmups_and_records'] if r['pair'] is not None]
    candidates=[r for r in timed if r['mode']=='new']
    prepare=statistics.median(r['cache_delta']['prepare_seconds'] for r in candidates)
    release=statistics.median(r['cache_delta']['release_seconds'] for r in candidates)
    peak=max(r['cache_peak_payload_bytes'] for r in candidates)/2**20
    allocated=candidates[0]['cache_delta']['allocated_bytes']/2**20
    outcome='通过>1秒速度条件，后续34/18题须另行记录。' if passed else '未超过1秒，按约定停止，不跑34例和18题，不采用或部署。'
    title='原4路＋缓存布局：完整请求实际省多少？'
    summary=f'3组完整请求已跑完。原路径中位{old:.3f}秒，布局候选{new:.3f}秒；每组再扣一次性安装后，配对净节省中位{net:+.3f}秒。{outcome}'
    head=['配对/顺序','原路径','原4路＋布局','完整请求差值','扣安装后的净节省']
    rows=[[f'{p["pair"]+1} / '+('原→新' if p['pair']%2==0 else '新→原'),f'{p["old_seconds"]:.6f}s',f'{p["new_seconds"]:.6f}s',f'{p["gross_saved_seconds"]:+.6f}s',f'{p["net_saved_seconds"]:+.6f}s'] for p in pairs]
    costhead=['项目','实际成本/边界']
    costs=[['请求内准备/复制/逐出（3次中位）',f'{prepare:.6f}s，已包含在候选请求耗时内'],
           ['请求结束清空（3次中位）',f'{release:.6f}s，已包含在候选请求耗时内'],
           ['一次性导入/安装',f'{install:.6f}s，保守地从每组差值全额扣除'],
           ['单请求缓存活跃峰值',f'{peak:.3f}MiB / 64MiB预算'],
           ['单请求累计派生分配',f'{allocated:.3f}MiB，是累计流量，非同时存活内存'],
           ['每次新请求', '72次构建 / 300次命中 / 62次逐出；结束缓存0'],
           ['全轮cgroup高水位',f'{data["whole_run_cgroup_peak_bytes"]/2**30:.3f}GiB / 8GiB，包含初始化/预热/检查；无OOM']]
    method='同一个Owner、原模型与12层，固定两问题B2/T1193（1193/1102，有效2295 token），批次与padding不变。原4路ELF没有重新编译；仅改成+64B行间隔的BF16副本，64MiB请求内LRU。选点、K连续求和顺序、线程和数值门槛不变。'
    timing='保留615/115保护预热，原/新各一次完整形状预热单列；随后原→新、新→原、原→新。每次候选从空缓存开始，计时包围完整system_one，包含复制、逐出和结束释放。请求计时内没有组件重放或SDK插桩，不挑最快值，不用历史31秒作基线。'
    numeric=f'{data["arm_checks"]}项ARM隔离检查通过；8次完整返回（2次形状预热＋6次计时）都与合格r072响应字典完全相同。对原CPU非confidence最大差{data["maximum_cpu_nonconfidence"]:.10g}，原0.01、choice/noul/usage/model门槛保持。每次372次修复、8,063,030选点，调用及修改计数一致，无拒绝、读取错误或CPU旁路。'
    memory='缓存结束后存活派生payload和条目均为0；RSS可能被分配器/NPU共享内存保留。cgroup峰值是整个容器的累计高水位，不能当作候选独占增量，不能与RSS相加。完整逐次RSS/cgroup快照在公开汇总JSON中。'
    limits='仅一条固定两问题开发请求的3组完整模型配对，不是HTTP延迟、完整官方榜单或全样本准确率，也不证明所有输入都能取得相同收益。r092约0.451秒是历史组件条件估算，本轮表中才是完整请求实测。'
    warmrows=[['原' if r['mode']=='old' else '新',f'{r["seconds"]:.6f}s'] for r in data['warmups_and_records'] if r['pair'] is None]
    memhead=['配对 / 路径','cgroup前→后','RSS前→后']
    memrows=[]
    for r in timed:
        values=[]
        for m in ('memory_before','memory_after'):
            snapshot=r[m]
            values.append((int(snapshot['memory.current'])/2**30,int(snapshot['process_status']['VmRSS'].split()[0])/2**20))
        memrows.append([f'{r["pair"]+1} / '+('原' if r['mode']=='old' else '新'),
                        f'{values[0][0]:.3f}→{values[1][0]:.3f}GiB',f'{values[0][1]:.3f}→{values[1][1]:.3f}GiB'])
    md=f'''# {title}

2026-10-20 · r093 · RK3576 · 仅实验，未部署

**{summary}**

## 3组完整请求配对

{table(head,rows)}

净节省＝同组原耗时−新耗时−一次性导入/安装；以这3个差值的中位数判断>1秒。原/新各自中位数之差不是配对中位数。

## 准备和内存都算进去

{table(costhead,costs)}

{memory}

请求前/后内存快照（包括原模型、共享内存和分配器，并非路径独占峰值）：

{table(memhead,memrows)}

## 数值与方法

{numeric}

{method}

{timing}

单列形状预热（不计入3组）：

{table(['路径','完整请求耗时'],warmrows)}

## 结论与范围

{outcome} 后端True/Torch3/highest/autocastFalse，所有句柄关闭、缓存与方法恢复检查通过；原同一服务容器已恢复healthy，控制器exit0，无超时或OOM。完整1,000条及定时任务继续暂停。

{limits}

[执行前固定的协议与源码](selected-layout-request-pairs/README.md) · [公开原始配对/成本/内存JSON](selected-layout-request-pairs/summary-r093.json) · [前轮组件估算](SELECTED_LAYOUT_TRIAL.html) · [耗时账本](REQUEST_COST.html)
'''
    (ROOT/'SELECTED_LAYOUT_REQUEST_PAIRS.md').write_text(md)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>原4路缓存布局：完整请求配对</title><style>body{margin:0;background:#f3f6fa;color:#1b2c40;font:16px/1.65 system-ui,sans-serif}main{max-width:1060px;margin:auto;padding:30px}h1{font-size:32px;line-height:1.25}h2{font-size:22px}.note{font-size:14px;color:#586b7c}.cards{display:flex;gap:14px;flex-wrap:wrap}.card,section{padding:22px;border-radius:14px;background:#fff;margin:20px 0}.card{flex:1;min-width:190px;margin:0}.card strong{display:block;font-size:28px;color:#176c63}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px 8px;border-bottom:1px solid #e6edf3;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}.bar{display:grid;grid-template-columns:80px 1fr 90px;gap:10px;align-items:center;margin:7px 0;font-size:14px}.track{background:#edf2f6;height:22px;border-radius:5px;overflow:hidden}.fill{height:100%;background:#65839d}.new{background:#258879}.pair{margin:20px 0}a{color:#2463ad}summary{cursor:pointer;font-weight:600}@media(max-width:650px){main{padding:16px}h1{font-size:27px}section{padding:16px}.bar{grid-template-columns:65px 1fr 80px}}</style><main>'''
    page+='<p class="note">r093 · 2026-10-20 · 完整请求实测 / 数值门槛不变</p><h1>'+title+'</h1><p>'+summary+'</p>'
    page+=f'<div class="cards"><div class="card">3组配对净节省中位<strong>{net:+.3f}秒</strong></div><div class="card">进入34/18题的条件<strong>严格 &gt;1秒</strong></div><div class="card">缓存活跃峰值<strong>{peak:.1f}MiB</strong></div></div>'
    page+='<section><h2>每一组，直接比完整请求</h2>'
    scale=max(p[k] for p in pairs for k in ('old_seconds','new_seconds'))*1.02
    for p in pairs:
        page+='<div class="pair"><b>第'+str(p['pair']+1)+'组 · '+('原→新' if p['pair']%2==0 else '新→原')+'</b>'
        for key,label,css in [('old_seconds','原路径',''),('new_seconds','原4路＋布局',' new')]:
            page+=f'<div class="bar"><span>{label}</span><div class="track"><div class="fill{css}" style="width:{p[key]/scale*100:.2f}%"></div></div><b>{p[key]:.3f}s</b></div>'
        page+='</div>'
    page+=htable(head,rows)+'<p class="note">上表候选完整耗时已含准备与释放；净节省再全额扣一次性安装。以3个配对差值中位数验收。</p></section>'
    page+='<section><h2>副本的时间和内存成本</h2>'+htable(costhead,costs)+'<details><summary>查看每次请求前后内存</summary>'+htable(memhead,memrows)+'<p>'+memory+'</p></details></section>'
    page+='<section><h2>数值没有放宽</h2><p>'+numeric+'</p><p class="note">'+method+'</p><details><summary>预热、计时与资源恢复</summary><p>'+timing+'</p>'+htable(['形状预热','完整请求耗时'],warmrows)+'<p>原服务同一容器恢复healthy，缓存/方法恢复、句柄关闭通过，控制器正常退出。无OOM或超时；未部署，完整1,000条及定时任务暂停。</p></details></section>'
    page+='<section><h2>下一步</h2><p>'+outcome+'</p><p class="note">'+limits+'</p><p><a href="SELECTED_LAYOUT_REQUEST_PAIRS.md">Markdown</a> · <a href="selected-layout-request-pairs/summary-r093.json">原始汇总JSON</a> · <a href="selected-layout-request-pairs/README.md">协议与源码</a> · <a href="SELECTED_LAYOUT_TRIAL.html">前轮组件结果</a></p></section></main></html>'
    (ROOT/'SELECTED_LAYOUT_REQUEST_PAIRS.html').write_text(page)
    print('Rendered SELECTED_LAYOUT_REQUEST_PAIRS.md/html')

if __name__=='__main__':main()
