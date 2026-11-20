"""Audit private PMU evidence and export aggregate geometry/counter summaries."""
import argparse,collections,hashlib,json,statistics
from pathlib import Path
from summarize import compare

def load(path):return json.loads(path.read_text())
def median(values):return statistics.median(values)
def audit(folder):
    stage=folder.parent;base=stage.parent;data=load(folder/'selected-dot-diagnosis.json');execution=load(folder/'execution.json')
    assert data['completed'] and len(data['records'])==528
    assert execution['exit_code']==0 and execution['request_completed'] and execution['retained_healthy'] and not execution['cleanup_errors']
    assert not execution.get('budget_exceeded') and not execution['worker_final_state']['OOMKilled']
    assert load(folder/'lab-sources.json')==load(stage/'candidate-sources.json')
    prior_sources=load(base/'r088/candidate-sources.json')['sources']
    sources=load(folder/'lab-sources.json')
    changes=[name for name,value in prior_sources.items() if name in sources and sources[name]!=value]
    assert set(changes)<=set(('probe.py','host_run.py','test_host_run.py')),changes
    assert hashlib.sha256(Path(__file__).with_name('check_regression.py').read_bytes()).hexdigest()=='841c6b517241d3d9af051bcd18a2f51a83c2d30a4aef4b05eaec1e85f1195683'
    prior_folder=next(p for p in (base/'r088').glob('*selected-dot-preflight*') if p.is_dir())
    prior=load(prior_folder/'selected-dot-preflight.json')['records']
    identity=lambda r:(r['index'],r['shape'],r['selected'],r['positions_sha256'])
    assert list(map(identity,data['capture_calls']))==list(map(identity,prior))
    assert len(prior)==372 and sum(r['selected'] for r in prior)==8063030
    assert len(data['captures'])==9
    assert all(r['input_alignment64']==r['replay_input_alignment64']==0 and r['weight_alignment64'] in (0,16) for r in data['captures'])
    qualified=load(base/'r072/20261020T004105-diagnostic-home-pair-full-batch-sparse-band25-plain/result.json')['response']
    cpu=load(base/'r061/20261019T170841-diagnostic-home-pair-cpu-full-plain/result.json')['response']
    assert data['response']==qualified
    comparison=compare(data['response'],cpu,False)
    assert comparison['max_delta']<=.01 and all(comparison[k] for k in ('choices_unchanged','noul_predictions_unchanged','usage_equal','model_equal'))
    backend=load(folder/'runtime-backend-before-cleanup.json')
    assert backend['mkldnn'] and backend['threads']==3 and backend['float32_precision']=='highest' and not backend['autocast_cpu']
    assert all(v==0 for v in load(folder/'lifecycle.json').values())
    tests=load(folder/'isolated-boundary-checks.json');assert tests['passed']
    groups=collections.defaultdict(list)
    for r in data['records']:
        pmu=r['pmu'];assert not pmu['thread_set_changed'] and pmu['min_running_fraction']>=.99
        assert pmu['totals']['cpu_cycles']>0 and pmu['totals']['inst_retired']>0
        groups[(tuple(r['shape']),r['layout'],r['suite'],r['mode'])].append(r)
    rows=[]
    for (shape,layout,suite,mode),records in groups.items():
        assert len(records)==4 and {r['round'] for r in records}==set(range(4))
        old={r['round']:r for r in groups[(shape,layout,suite,'old')]}
        keys=records[0]['pmu']['totals']
        counts={k:median(r['pmu']['totals'][k]/r['repeats']/r['selected'] for r in records) for k in keys}
        row={'shape':list(shape),'layout':layout,'suite':suite,'mode':mode,'median_ms_per_call':1000*median(r['seconds_per_call'] for r in records),
             'paired_time_ratio_to_old':median(r['seconds_per_call']/old[r['round']]['seconds_per_call'] for r in records),
             'counts_per_selected':counts,'median_cpi':median(r['pmu']['totals']['cpu_cycles']/r['pmu']['totals']['inst_retired'] for r in records),
             'median_active_threads':median(r['pmu']['active_threads'] for r in records),
             'paired_time_ratios':[r['seconds_per_call']/old[r['round']]['seconds_per_call'] for r in records]}
        rows.append(row)
    result={'passed':True,'windows':len(data['records']),'calls':372,'selected':8063030,'geometries':9,'workloads':11,
            'arm_tests':tests['tests_run'],'changed_prior_files':changes,'vs_original_cpu':comparison,
            'minimum_running_fraction':min(r['pmu']['min_running_fraction'] for r in data['records']),
            'all_replay_bf16_bits_equal':True,'rows':rows,'padding_alignment_confound':'Original weights in the two controls are offset16 modulo64, whereas pad64 allocations are aligned0; requires separate factorial control before attributing stride versus alignment.',
            'scope':'Frozen-kernel CPU replay, not candidate model latency; counts are target-process thread sums, not additive wall time.'}
    (stage/'diagnosis-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for r in rows:
        if r['suite']=='core':print(r['shape'],r['layout'],r['mode'],'ms',round(r['median_ms_per_call'],3),'ratio',round(r['paired_time_ratio_to_old'],3),'cpi',round(r['median_cpi'],3),'L1refill/point',round(r['counts_per_selected']['l1d_cache_refill'],2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);audit(p.parse_args().folder)
