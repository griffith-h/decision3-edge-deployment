"""Development-case parity against the previously completed native response."""
import argparse,json,statistics
from pathlib import Path
from summarize import compare

def check(folder):
    execution=json.loads((folder/'execution.json').read_text())
    assert execution['exit_code']==0 and execution['retained_healthy'] and not execution.get('cleanup_errors')
    data=json.loads((folder/'result.json').read_text());assert data['completed']
    reference=json.loads((folder.parent/'references.json').read_text())[data['case']]['response']
    rows=[]
    for record in data['records']:
        assert record.get('encoder_restored',True),'Public encoder was not restored'
        parity=compare(record['response'],reference,False)
        passed=parity['max_delta']<=.01 and all(parity[k] for k in ('choices_unchanged','usage_equal','model_equal'))
        rows.append({'seconds':record['seconds'],'parity_without_confidence':parity,
                     'max_delta_including_confidence':compare(record['response'],reference)['max_delta'],'passed':passed})
    lifecycle=json.loads((folder/'lifecycle.json').read_text());assert all(v==0 for v in lifecycle.values())
    transition_path=folder/'after-long-transition.json'
    transitions=[]
    if data.get('long_readout_counts',{}).get('optimized',0):
        assert transition_path.exists(),'Missing actual private-readout long to short transition check'
    if transition_path.exists():
        transition=json.loads(transition_path.read_text());assert transition['completed']
        refs=json.loads((folder.parent/'regression-references.json').read_text())
        assert [r['id'] for r in transition['records']]==['protect615','protect115']
        for row in transition['records']:
            assert row['encoder_restored']
            key='words-512' if row['id']=='protect615' else 'single'
            parity=compare(row['response'],refs[key]['native'])
            assert parity['max_delta']<=1e-6 and all(parity[k] for k in ('choices_unchanged','usage_equal','model_equal'))
            transitions.append({'id':row['id'],'seconds':row['seconds'],'parity':parity})
    projection={}
    if data.get('long_projection_counts',{}).get('scopes',0):
        before=data['counts_before']['projections']['native_rows']
        after=data['counts_after']['projections']['native_rows']
        delta={k:v-before.get(k,0) for k,v in after.items() if v>before.get(k,0)}
        assert delta and max(map(int,delta))<=512,'New long projection exceeded native row bound'
        projection={'current_native_rows':delta,'max_native_rows':max(map(int,delta))}
    result={'passed':all(r['passed'] for r in rows),'case':data['case'],'profile':data['profile'],'records':rows,
            'median_seconds':statistics.median(r['seconds'] for r in rows),
            'projection_bounds':projection,'after_close':lifecycle,'after_long_transitions':transitions,
            'scope':'Fixed development request vs saved native response; numerical parity, not full-suite accuracy'}
    (folder/'case-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args()
    assert check(Path(a.folder))['passed'],'Development numerical gate failed'
