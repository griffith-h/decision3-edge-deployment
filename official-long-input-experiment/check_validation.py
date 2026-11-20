"""Check numerical, transition and paired gates of one development suite."""
import argparse,json
from pathlib import Path
from summarize import compare
from validation_suite import ORDER
from check_regression import check as check_regression
from check_protection import check as check_protection
from check_tools import check as check_tools

def check(folder,tools_reference):
    execution=json.loads((folder/'execution.json').read_text())
    assert execution['exit_code']==0 and execution['request_completed'] and execution['retained_healthy'] and not execution['cleanup_errors']
    data=json.loads((folder/'validation.json').read_text())
    assert data['completed'] and [r['id'] for r in data['records']]==list(ORDER)
    refs=json.loads((folder.parent/'references.json').read_text())
    regression_refs=json.loads((folder.parent/'regression-references.json').read_text())
    inputs={r['id']:r['payload'] for r in json.loads((folder.parent/'inputs.json').read_text())}
    regression_inputs=json.loads((folder.parent/'regression-inputs.json').read_text())
    checks=[]
    for record in data['records']:
        key=record['id'];assert record.get('encoder_restored',True),'Public encoder was not restored'
        if key=='tools32':continue
        if key.startswith('protect'):
            matches=[r['id'] for r in regression_inputs if {k:r['payload'][k] for k in ('state','questions')}==inputs[key]
                     and not r['payload'].get('images') and not r['payload'].get('videos')]
            assert matches
            expected=regression_refs[matches[0]]['native'];limit=1e-6
        else:expected=refs[key]['response'];limit=.01
        parity=compare(record['response'],expected,False)
        passed=parity['max_delta']<=limit and all(parity[k] for k in ('choices_unchanged','usage_equal','model_equal'))
        projection_rows={}
        if key in ('multi4605','multi4597'):
            before=record['counts_before']['projections']['native_rows'];after=record['counts_after']['projections']['native_rows']
            projection_rows={k:v-before.get(k,0) for k,v in after.items() if v>before.get(k,0)}
            assert projection_rows and max(map(int,projection_rows))<=512
        checks.append({'id':key,'seconds':record['seconds'],'numeric_without_confidence':parity,'limit':limit,
                       'max_delta_including_confidence':compare(record['response'],expected)['max_delta'],
                       'projection_rows':projection_rows,'passed':passed})
    regression=check_regression(folder);protection=check_protection(folder)
    tools=check_tools(folder/'tools-result.json',tools_reference)
    lifecycle=json.loads((folder/'lifecycle.json').read_text());assert all(v==0 for v in lifecycle.values())
    passed=all(c['passed'] for c in checks) and regression['passed'] and protection['passed'] and tools['passed']
    result={'passed':passed,'development_calls':len(data['records']),'checks':checks,
            'regression34_passed':regression['passed'],'protection_passed':protection['passed'],
            'tools32_passed':tools['passed'],'after_close':lifecycle,
            'scope':'Development numerical parity and warm paired checks, not full-suite accuracy or P95'}
    (folder/'validation-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('tools_reference');a=p.parse_args()
    assert check(Path(a.folder),Path(a.tools_reference))['passed'],'Development suite failed'
