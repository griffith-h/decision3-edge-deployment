"""Audit fixed layout/tail candidate before paying for full-model pairing."""
import argparse,collections,hashlib,json
from pathlib import Path
from summarize import compare

def load(p):return json.loads(p.read_text())
def audit(folder):
    stage=folder.parent;base=stage.parent;data=load(folder/'layout-preflight.json');e=load(folder/'execution.json')
    assert data['completed'] and e['exit_code']==0 and e['request_completed'] and e['retained_healthy'] and not e['cleanup_errors'] and not e.get('budget_exceeded') and not e['worker_final_state']['OOMKilled']
    sources=load(folder/'lab-sources.json');assert sources==load(stage/'candidate-sources.json')
    oldpin=load(base/'r089/candidate-sources.json');assert all(sources[k]==v for k,v in oldpin.items() if k.endswith(('.so','.c','.cpp')))
    assert hashlib.sha256(Path(__file__).with_name('check_regression.py').read_bytes()).hexdigest()=='841c6b517241d3d9af051bcd18a2f51a83c2d30a4aef4b05eaec1e85f1195683'
    prior=load(next((base/'r088').glob('2026*/selected-dot-preflight.json')))['records']
    identity=lambda r:(r['index'],r['shape'],r['selected'],r['positions_sha256'])
    assert list(map(identity,data['records']))==list(map(identity,prior))
    assert len(prior)==372 and sum(r['selected'] for r in prior)==8063030 and len(data['benchmarks'])==9
    expected_modes={'original4_layout'} if stage.name=='r092' else {'layout_only','tail_only','combined'}
    assert all(set(r['modes'])==expected_modes and all(all(v.values()) for v in r['modes'].values()) for r in data['records'])
    cpu=load(base/'r061/20261019T170841-diagnostic-home-pair-cpu-full-plain/result.json')['response'];qualified=load(base/'r072/20261020T004105-diagnostic-home-pair-full-batch-sparse-band25-plain/result.json')['response']
    assert data['response']==qualified;c=compare(data['response'],cpu,False)
    assert c['max_delta']<=.01 and all(c[k] for k in ('choices_unchanged','noul_predictions_unchanged','usage_equal','model_equal'))
    backend=load(folder/'runtime-backend-before-cleanup.json');assert backend['mkldnn'] and backend['threads']==3 and backend['float32_precision']=='highest' and not backend['autocast_cpu']
    assert all(v==0 for v in load(folder/'lifecycle.json').values());tests=load(folder/'isolated-boundary-checks.json');assert tests['passed']
    totals=collections.Counter()
    for r in data['records']:totals[tuple(r['shape'])]+=r['selected']
    modes=[]
    for mode in sorted(expected_modes):
        old=new=0.;shapes=[]
        for b in data['benchmarks']:
            item=b['modes'][mode];assert len(item['pairs'])==3;scale=totals[tuple(b['shape'])]/b['selected']
            old+=scale*item['median_old'];new+=scale*item['median_new'];shapes.append({'shape':b['shape'],'ratio':item['median_new']/item['median_old']})
        cache=data['caches'][mode];assert cache['live_bytes']==cache['entries']==0 and cache['counts']['peak_bytes']<=64*1024*1024
        assert cache['counts']['calls']==399 and cache['counts']['selected']==8612642
        if mode!='tail_only':assert cache['counts']['builds']==len(data['unique_weight_shapes'])==72
        overhead=cache['counts']['prepare_seconds']+cache['counts']['release_seconds'];ratio=new/old;prediction=5.679774793195769*(1-ratio)-overhead
        modes.append({'mode':mode,'weighted_component_ratio':ratio,'observed_prepare_release_seconds':overhead,'estimated_net_request_saving_seconds':prediction,'potential_above_1s':prediction>1.,'shapes':shapes,'cache_counts':cache['counts']})
    primary='original4_layout' if stage.name=='r092' else 'combined'
    combined=next(r for r in modes if r['mode']==primary)
    result={'passed_numerical':True,'arm_checks':tests['tests_run'],'calls':372,'selected':8063030,'geometries':9,'unique_weights':len(data['unique_weight_shapes']),'source_bf16_bytes':sum(w['bf16_bytes'] for w in data['unique_weight_shapes']),'memory_before':data['memory_before'],'memory_after':data['memory_after'],'geometry_selected_counts':[{'shape':list(k),'selected':v} for k,v in totals.items()],'component_benchmarks':data['benchmarks'],'modes':modes,'primary_mode':primary,'primary_eligible_for_request_pairs':combined['potential_above_1s'],'vs_cpu':c,'ablation_limit':('Original qualified4lane binary plus layout only' if stage.name=='r092' else 'layout_only uses new4lane helper, not original ELF; tail_only includes wider main groups as well as tail dispatch. r092 isolates unchanged original4 plus layout.'),'scope':'Weighted representative component estimate minus observed preparation/release only; initialization and real end-to-end must pass full request pairs, not yet a speed result.'}
    (stage/'layout-preflight-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);audit(p.parse_args().folder)
