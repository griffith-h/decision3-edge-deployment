import sys
sys.path.insert(0,'/long')
import baseline
import tail_native as n
import long_kernels as lk
import torch,time,json,statistics
from pathlib import Path
torch.set_num_threads(3);P=Path('/experiment');captured=torch.load(P/'captured.pt',weights_only=False)
r={'attention':{},'scan':{},'scope':'6 captured real layer inputs, three rotating repeats; no model or answer reuse in end-to-end tests'}
configs={'attention':{'baseline':None,**{f'q{tile}t{threads}r{recip}':{'tile':tile,'threads':threads,'reciprocal':recip} for tile in (32,64,128,256,615) for threads in (2,3) for recip in (1,)},'q64t3r0':{'tile':64,'threads':3,'reciprocal':0}},'scan':{'baseline':None,**{f'b{block}v{tile}t{threads}':{'kind':'block','block':block,'tile':tile,'threads':threads} for block in (16,32,64,128) for tile in (32,64,128) for threads in (2,3)},**{f'r{mode}t{threads}':{'kind':'recurrent','mode':mode,'threads':threads} for mode in (6,7) for threads in (2,3)}}}
def save():(P/'logs/replay.json').write_text(json.dumps(r,indent=2)+'\n')
with torch.inference_mode():
    for kind in ('attention','scan'):
        for name,cfg in configs[kind].items():r[kind][name]={'config':cfg,'records':[],'max_abs':0.,'passed':True}
        for layer,c in enumerate(captured[kind]):
            for i in range(3):
                names=list(configs[kind]);shift=(i+layer)%len(names);names=names[shift:]+names[:shift]
                for name in names:
                    cfg=configs[kind][name];fn=(lk.attention if kind=='attention' else lk.scan_prepared) if cfg is None else (lambda *args,cfg=cfg: (n.attention if kind=='attention' else n.scan)(*args,**cfg))
                    t=time.perf_counter();out=fn(*c['args']);dt=time.perf_counter()-t;delta=(out.float()-c['reference'].float()).abs().max().item();passed=torch.allclose(out.float(),c['reference'].float(),atol=.002,rtol=.008)
                    v=r[kind][name];v['records'].append({'layer':layer,'seconds':dt,'max_abs':delta,'passed':passed});v['max_abs']=max(delta,v['max_abs']);v['passed']&=passed
            save();print('REPLAY',kind,layer+1,flush=True)
        for name,v in r[kind].items():v['sum_layer_medians']=sum(statistics.median(x['seconds'] for x in v['records'] if x['layer']==layer) for layer in range(6))
        r[kind+'_ranking']=sorted(r[kind],key=lambda name:(not r[kind][name]['passed'],r[kind][name]['sum_layer_medians']));save();print('RANK',kind,[(k,r[kind][k]['sum_layer_medians'],r[kind][k]['max_abs'],r[kind][k]['passed']) for k in r[kind+'_ranking'][:6]],flush=True)
r['completed']=True;save()
selected={kind:r[kind][next(name for name in r[kind+'_ranking'] if name!='baseline' and r[kind][name]['passed'])]['config'] for kind in ('attention','scan')}
(P/'selected.json').write_text(json.dumps(selected,indent=2)+'\n');print('SELECTED',selected,flush=True)
