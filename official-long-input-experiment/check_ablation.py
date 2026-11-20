"""Review isolated components against an actual original-CPU B2 forward."""
import argparse,hashlib,json
from pathlib import Path
from summarize import compare

MODES=('candidate','cpu_scan','cpu_attention','cpu_mlp','cpu_projection',
       'all_four_cpu','original_current_cpu_full_pair')
SINGLE_MODES=('only_scan','only_attention','only_mlp','only_projection','original_current_cpu_full_pair')

def check(folder,original_reference):
    execution=json.loads((folder/'execution.json').read_text())
    assert execution['request_completed'] and execution['exit_code']==0 and execution['retained_healthy'] and not execution['cleanup_errors']
    data=json.loads((folder/'ablation.json').read_text())
    expected_modes=SINGLE_MODES if data.get('suite') in ('single_enable','single_observe') else MODES
    assert data['completed'] and [r['mode'] for r in data['records']]==list(expected_modes)
    gold=data['records'][-1];assert gold['npu_calls_delta']==[0,0,0]
    records=[]
    for row in data['records']:
        assert row['encoder_restored']
        a,b=row['counts_after'],row['counts_before']
        hits={'projection':a['projections']['optimized_projections']-b['projections']['optimized_projections'],
              'mlp':a['native_mlp_calls']-b['native_mlp_calls'],
              **{key:a['long_kernels'][key]['optimized']-b['long_kernels'][key]['optimized'] for key in ('scan','attention')}}
        disabled=list(hits) if row['mode'] in ('all_four_cpu','original_current_cpu_full_pair') else [row['mode'][4:]] if row['mode'].startswith('cpu_') else []
        if row['mode'].startswith('only_'):disabled=[k for k in hits if k!=row['mode'][5:]]
        assert all(hits[k]==0 for k in disabled),(row['mode'],'Ablation did not bypass intended component',hits)
        records.append({'mode':row['mode'],'seconds_diagnostic_only':row['seconds'],
                        'vs_actual_cpu':compare(row['response'],gold['response'],False),
                        'max_including_confidence':compare(row['response'],gold['response'])['max_delta'],
                        'native_component_hits':hits})
    old=json.loads(original_reference.read_text());assert old['completed'] and old['npu_matrix_calls']==0
    metadata=json.loads((folder.parent/'diagnostic-pair-manifest.json').read_text())
    keys=list(old['response']['answers']);selected=[keys[i] for i in metadata['question_indices']]
    tokens=sum(old['reference_shapes'][i]['effective_tokens'] for i in metadata['question_indices'])
    assert all(old['reference_shapes'][i]['original_batch_width']==1193 for i in metadata['question_indices'])
    expected={**old['response'],'answers':{k:old['response']['answers'][k] for k in selected},
              'usage':{'input_tokens':tokens,'output_tokens':0}}
    oracle=compare(gold['response'],expected,False)
    lifecycle=json.loads((folder/'lifecycle.json').read_text());assert all(v==0 for v in lifecycle.values())
    result={'scope':'Two frozen diagnostic questions; no benchmark-group metric or final performance claim',
            'actual_cpu_vs_independent_original_width_rows':oracle,'records':records,'after_close':lifecycle,
            'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (folder/'ablation-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('original_reference');a=p.parse_args()
    check(Path(a.folder),Path(a.original_reference))
