import argparse,json,statistics
from pathlib import Path
from summarize import compare

def check(folder):
    execution=json.loads((folder/'execution.json').read_text())
    assert execution['exit_code']==0 and execution['retained_healthy'] and not execution.get('cleanup_errors')
    data=json.loads((folder/'protection.json').read_text());assert data['completed'] and len(data['records'])==20
    results={}
    for case in ('protect115','protect615'):
        rows=[r for r in data['records'] if r['case']==case];assert len(rows)==10
        paired=[];base=[];current=[];max_delta=0.
        for start in range(0,10,2):
            a=next(r for r in rows[start:start+2] if not r['candidate']);b=next(r for r in rows[start:start+2] if r['candidate'])
            parity=compare(b['response'],a['response'])
            assert parity['max_delta']<=1e-6 and all(parity[k] for k in ('choices_unchanged','usage_equal','model_equal'))
            max_delta=max(max_delta,parity['max_delta']);base.append(a['seconds']);current.append(b['seconds']);paired.append(b['seconds']/a['seconds'])
        ratio=statistics.median(paired)
        results[case]={'pairs':5,'baseline_median_ms':statistics.median(base)*1000,'candidate_median_ms':statistics.median(current)*1000,'median_paired_ratio':ratio,'max_numeric_delta_including_confidence':max_delta,'passed':ratio<=1.1}
    result={'passed':all(x['passed'] for x in results.values()),'cases':results,'scope':'Five warm paired method calls per protected shape, no P95/SLA claim'}
    (folder/'protection-check.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();assert check(Path(a.folder))['passed']
