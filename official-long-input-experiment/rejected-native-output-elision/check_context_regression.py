"""Freeze independent same-context references, apply unchanged regression gates.

Historical outputs are reported separately and never overwritten. New reference
responses come only from original CPU/native owners before candidate execution.
"""
import argparse,hashlib,json
from pathlib import Path
from context_regression import EXPECTED,require_backend
from check_regression import check_pair
from summarize import compare

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read_run(folder,variant):
    execution=json.loads((folder/'execution.json').read_text())
    assert execution['exit_code']==0 and execution['request_completed'] and execution['retained_healthy'] and not execution['cleanup_errors']
    assert not execution.get('budget_exceeded') and not execution['worker_final_state']['OOMKilled']
    path=folder/'context-regression.json';data=json.loads(path.read_text())
    assert data['completed'] and data['variant']==variant and data['context']==EXPECTED
    assert len(data['records'])==34 and [r['id'] for r in data['records']]==data['case_ids'] and len(set(data['case_ids']))==34
    inputs=folder.parent/'regression-inputs.json';expected=json.loads(inputs.read_text())
    assert digest(inputs)==data['input_sha256'] and [r['id'] for r in expected]==data['case_ids']
    expected_payloads={r['id']:r['payload'] for r in expected}
    for row in data['records']:
        require_backend(row['backend_before']);require_backend(row['backend_after'])
        assert set(row['response']['answers'])==set(expected_payloads[row['id']]['questions'])
        assert not any('error' in value for value in row['response']['answers'].values())
        if variant=='reference':assert not any(row['npu_calls_delta'])
        if variant=='baseline' and row['response']['usage'].get('input_tokens')==615:assert row['npu_calls_delta'][1]>0
        if variant=='baseline' and row['response']['usage'].get('input_tokens')==115:assert row['npu_calls_delta'][3]>0
    require_backend(json.loads((folder/'runtime-backend-before-cleanup.json').read_text()))
    assert all(v==0 for v in json.loads((folder/'lifecycle.json').read_text()).values())
    checks=folder/('isolated-boundary-checks.json' if variant.startswith('full-batch') else 'isolated-request-scope-checks.json')
    assert json.loads(checks.read_text())['passed']
    return data

def freeze(cpu_folder,native_folder,target):
    assert not target.exists(),'Do not overwrite frozen context references'
    cpu=read_run(cpu_folder,'reference');native=read_run(native_folder,'baseline')
    assert cpu['case_ids']==native['case_ids'] and cpu['input_sha256']==native['input_sha256']
    assert (cpu_folder/'identity.json').read_bytes()==(native_folder/'identity.json').read_bytes()
    assert (cpu_folder/'lab-sources.json').read_bytes()==(native_folder/'lab-sources.json').read_bytes()
    def response_map(data):return {r['id']:r['response'] for r in data['records']}
    c,n=map(response_map,(cpu,native))
    refs={k:{'current':c[k],'native':n[k]} for k in cpu['case_ids']}
    data={'context':EXPECTED,'case_ids':cpu['case_ids'],'input_sha256':cpu['input_sha256'],
        'sources':{str(p/'context-regression.json'):digest(p/'context-regression.json') for p in (cpu_folder,native_folder)},
        'identity_sha256':digest(cpu_folder/'identity.json'),'lab_sources_sha256':digest(cpu_folder/'lab-sources.json'),
        'references':refs,'thresholds_unchanged':{'cpu_native_nonconfidence':.01,'protected_native_all_numeric':1e-6},
        'candidate_not_used':True,'checker_sha256':digest(Path(__file__)),'gate_source_sha256':digest(Path(__file__).with_name('check_regression.py'))}
    target.write_text(json.dumps(data,indent=2)+'\n');return data

def check(folder,reference_file,historical_file,expected_variant='full-batch-sparse-request-scope',candidate_sources=None):
    assert expected_variant in ('full-batch-sparse-request-scope','full-batch-sparse-no-output')
    data=read_run(folder,expected_variant);refs=json.loads(reference_file.read_text());history=json.loads(historical_file.read_text())
    assert data['case_ids']==refs['case_ids'] and data['input_sha256']==refs['input_sha256']
    assert digest(Path(__file__).with_name('check_regression.py'))==refs['gate_source_sha256'],'Original numerical gate source changed'
    assert digest(folder/'identity.json')==refs['identity_sha256']
    if candidate_sources is None:assert digest(folder/'lab-sources.json')==refs['lab_sources_sha256']
    else:
        pinned=json.loads(candidate_sources.read_text())
        assert pinned['variant']==expected_variant and pinned['references_sha256']==digest(reference_file)
        assert json.loads((folder/'lab-sources.json').read_text())==pinned['sources']
    rows=[]
    for row in data['records']:
        key=row['id'];failure=None
        try:check_pair(row['response'],refs['references'][key])
        except AssertionError as exc:failure=str(exc)
        old_failure=None
        try:check_pair(row['response'],history[key])
        except AssertionError as exc:old_failure=str(exc)
        # Retain the old CPU/native0.01 and discrete gates as well. Only the
        # strict identity comparison needs the audited same-backend native.
        historical_core=[compare(row['response'],history[key][role],False) for role in ('current','native')]
        historical_core_passed=all(p['max_delta']<=.01 and all(p[k] for k in ('choices_unchanged','usage_equal','model_equal')) for p in historical_core)
        rows.append({'id':key,'passed':failure is None and historical_core_passed,'failure':failure,
            'historical_core_passed':historical_core_passed,'seconds':row['seconds'],
            'vs_current':compare(row['response'],refs['references'][key]['current'],False),
            'vs_current_all_numeric':compare(row['response'],refs['references'][key]['current']),
            'vs_native':compare(row['response'],refs['references'][key]['native'],False),
            'vs_native_all_numeric':compare(row['response'],refs['references'][key]['native']),
            'historical_check_passed':old_failure is None,'historical_failure':old_failure,
            'historical_current':compare(row['response'],history[key]['current'],False),
            'historical_native_all_numeric':compare(row['response'],history[key]['native'])})
    result={'passed':all(r['passed'] for r in rows),'passed_cases':sum(r['passed'] for r in rows),'cases':34,
        'context':EXPECTED,'variant':expected_variant,'candidate_sources_sha256':digest(candidate_sources) if candidate_sources else None,'checker_sha256':digest(Path(__file__)),'gate_source_sha256':digest(Path(__file__).with_name('check_regression.py')),'references_sha256':digest(reference_file),'historical_references_sha256':digest(historical_file),
        'gate':'Unchanged check_regression.check_pair: bothCPU/native nonconfidence<=0.01, choice/noul/usage/model equal,115/615 native allnumeric<=1e-6;confidence separate; historicalCPU/native0.01 and discrete gates also retained',
        'maximum_cpu_nonconfidence':max(r['vs_current']['max_delta'] for r in rows),
        'maximum_cpu_all_numeric':max(r['vs_current_all_numeric']['max_delta'] for r in rows),
        'rows':rows,'scope':'34 self-built regression cases, not official accuracy or full1000 acceptance'}
    target=folder/'context-regression-check.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    f=sub.add_parser('freeze');f.add_argument('cpu',type=Path);f.add_argument('native',type=Path);f.add_argument('target',type=Path)
    c=sub.add_parser('check');c.add_argument('candidate',type=Path);c.add_argument('references',type=Path);c.add_argument('historical',type=Path)
    a=p.parse_args()
    if a.action=='freeze':r=freeze(a.cpu,a.native,a.target);print('Frozen independent34 context references')
    else:
        r=check(a.candidate,a.references,a.historical);print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2));assert r['passed'],'Full34 unchanged gate failed'
