"""Numerical and native ranking checks on this development query only."""
import argparse,json,statistics
from pathlib import Path
from decision_index.suite.io import read_jsonl
from decision_index.scoring.metrics import score_query
from summarize import compare,numbers

def check(candidate_path,reference_path):
    candidate=json.loads(candidate_path.read_text());reference=json.loads(reference_path.read_text())
    for folder in (candidate_path.parent,reference_path.parent):
        execution=json.loads((folder/'execution.json').read_text())
        assert execution['exit_code']==0 and execution['retained_healthy'] and not execution.get('cleanup_errors')
    assert candidate['completed'] and reference['completed'] and reference['npu_matrix_calls']==0
    actual,expected=candidate['response'],reference['response']
    assert len(actual['answers'])==len(expected['answers'])==32
    assert not any('error' in v for v in actual['answers'].values())
    numerical=compare(actual,expected,False);all_numeric=compare(actual,expected,True)
    ca,cb=numbers(actual['answers']),numbers(expected['answers'])
    confidence=max((abs(ca[k]-cb[k]) for k in ca if k.endswith('/confidence')),default=0)
    root=Path('output/decision-index-audit/iterations/r001')
    manifest=json.loads((root/'cases-manifest.json').read_text())
    run_id=next(c['run_id'] for c in manifest['cases'] if c['id']=='tools32')
    sample=list(read_jsonl('output/decision-index-audit/sample-1000.jsonl.gz'))
    row=next(r for r in sample if r['_evaluation']['run_id']==run_id)
    group=[r for r in sample if (r['_evaluation']['catalog_id'],r['_evaluation']['group_id'])==(row['_evaluation']['catalog_id'],row['_evaluation']['group_id'])]
    assert len(group)==1,'Ranking group spans other rows: cannot score this request alone'
    mapping=row['scoring']['field_to_document']
    def native(response):return score_query(row['scoring'],{doc:response['answers'][key]['probabilities']['yes'] for key,doc in mapping.items()})
    score,ref_score=native(actual),native(expected)
    assert score['valid'] and ref_score['valid']
    passed=(numerical['max_delta']<=.01 and all(numerical[k] for k in ('choices_unchanged','usage_equal','model_equal')) and score['ndcg_at_10']>=ref_score['ndcg_at_10']-1e-12)
    repetitions=[]
    for record in candidate.get('records',[candidate]):
        parity=compare(record['response'],expected,False)
        task=native(record['response'])
        repeat_pass=(parity['max_delta']<=.01 and all(parity[k] for k in ('choices_unchanged','usage_equal','model_equal'))
                     and task['valid'] and task['ndcg_at_10']>=ref_score['ndcg_at_10']-1e-12)
        repetitions.append({'seconds':record['seconds'],'parity':parity,'task_metrics':task,'passed':repeat_pass})
        passed=passed and repeat_pass
    result={'passed':passed,'scope':'one complete official-source development query, 32 ranked documents; not full-suite accuracy',
            'numeric_without_confidence':numerical,'confidence_max_delta':confidence,'all_numeric_max_delta':all_numeric['max_delta'],
            'candidate_task_metrics':score,'cpu_reference_task_metrics':ref_score,'input_tokens':actual['usage']['input_tokens'],
            'repetitions':repetitions,'median_seconds':statistics.median(r['seconds'] for r in repetitions)}
    output=candidate_path.with_name('tools-check.json');output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('reference');a=p.parse_args()
    assert check(Path(a.candidate),Path(a.reference))['passed'],'New query numerical or native-score gate failed'
