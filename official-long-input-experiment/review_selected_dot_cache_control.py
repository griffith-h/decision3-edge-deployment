"""Validate factorial cache controls without exporting captured tensors or labels."""
import argparse,collections,json,statistics
from pathlib import Path

def load(p):return json.loads(p.read_text())
def med(x):return statistics.median(x)
def audit(folder):
    stage=folder.parent;base=stage.parent;data=load(folder/'cache-control.json');execution=load(folder/'execution.json')
    assert data['completed'] and len(data['records'])==512
    assert execution['exit_code']==0 and execution['request_completed'] and execution['retained_healthy'] and not execution['cleanup_errors']
    assert not execution.get('budget_exceeded') and not execution['worker_final_state']['OOMKilled']
    sources=load(folder/'lab-sources.json');assert sources==load(stage/'candidate-sources.json')
    old_sources=load(base/'r089/candidate-sources.json');changes=[k for k,v in old_sources.items() if k in sources and sources[k]!=v]
    assert set(changes)<=set(('probe.py','host_run.py','test_host_run.py')),changes
    previous=next(p for p in (base/'r089').glob('*selected-dot-diagnosis-*') if p.is_dir())
    captures={c['index']:c for c in load(previous/'selected-dot-diagnosis.json')['captures']}
    for c in data['captures']:
        old=captures[c['index']]
        assert c['sha256']==old['capture_file_sha256'] and c['positions_sha256']==old['positions_sha256'] and c['shape']==old['shape'] and c['selected']==old['selected']
    groups=collections.defaultdict(list)
    for r in data['records']:
        assert not r['pmu']['thread_set_changed'] and r['pmu']['min_running_fraction']>=.99
        assert r['pmu']['totals']['cpu_cycles']>0 and r['pmu']['totals']['inst_retired']>0
        groups[(tuple(r['shape']),r['suite'],r['padding_bytes'],r['alignment64'],r['mode'])].append(r)
    rows=[]
    for (shape,suite,padding,alignment,mode),records in groups.items():
        assert len(records)==8 and {r['round'] for r in records}==set(range(8))
        baseline={r['round']:r for r in groups[(shape,suite,0,alignment,mode)]}
        paired=[r['seconds_per_call']/baseline[r['round']]['seconds_per_call'] for r in records]
        counts={k:med(r['pmu']['totals'][k]/r['repeats']/r['selected'] for r in records) for k in records[0]['pmu']['totals']}
        rows.append({'shape':list(shape),'suite':suite,'padding_bytes':padding,'alignment64':alignment,'mode':mode,
            'median_ms_per_call':1000*med(r['seconds_per_call'] for r in records),
            'counts_per_selected':counts,'median_cpi':med(r['pmu']['totals']['cpu_cycles']/r['pmu']['totals']['inst_retired'] for r in records),
            'paired_pad_ratio_same_alignment':med(paired),'paired_pad_ratios_same_alignment':paired,
            'pad_faster_rounds':sum(v<1 for v in paired),
            'involuntary_context_switches_total':sum(r['involuntary_context_switches'] for r in records)})
    states=collections.defaultdict(set)
    for r in data['records']:
        for name in ('state_before','state_after'):
            for k,v in r[name].items():states[k].add(v)
    result={'passed':True,'windows':512,'all_bf16_bits_equal':True,'rounds':8,'shapes':2,'rows':rows,
        'minimum_running_fraction':min(r['pmu']['min_running_fraction'] for r in data['records']),
        'observed_frequencies':{k:sorted(v) for k,v in states.items() if 'freq' in k},
        'changed_prior_files':changes,'scope':'Factorial CPU replay with copied real input values and fixed binaries, not whole-model speed. All timing observations retained, including scheduler noise.'}
    (stage/'cache-control-review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    for r in rows:
        if r['suite']=='core':print(r['shape'],r['padding_bytes'],r['alignment64'],r['mode'],'ms',round(r['median_ms_per_call'],3),'pad_ratio',round(r['paired_pad_ratio_same_alignment'],3),'L1refill',round(r['counts_per_selected']['l1d_cache_refill'],2))
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);audit(p.parse_args().folder)
