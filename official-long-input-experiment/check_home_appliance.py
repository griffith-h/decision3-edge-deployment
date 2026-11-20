"""Numerical and upstream case/field accuracy for the complete development case.

Only catalog9's single-request complete group is handled here. No full-suite
accuracy claim; CPU reference is correctness-only, never a speed baseline.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
from decision_index.suite.io import read_jsonl,sha256_file
from decision_index.scoring.report import score
from summarize import compare

def checked(path):
    execution=json.loads((path.parent/'execution.json').read_text())
    assert execution['exit_code']==0 and execution['retained_healthy'] and not execution['cleanup_errors']
    data=json.loads(path.read_text());assert data['completed']
    return data

def check(candidate_path,reference_path,parent_path,output=None):
    candidate,reference,parent=map(checked,(candidate_path,reference_path,parent_path))
    assert reference['npu_matrix_calls']==0 and candidate['case']==reference['case']==parent['case']
    if reference['variant']=='cpu-full':
        assert [(len(r['lengths']),r['width']) for r in reference['records']]==[(8,1181),(8,1193),(2,1105)]
        assert all(r['npu_calls_delta']==[0,0,0,0] for r in reference['records'])
        assert reference['response']['usage']['input_tokens']==20051
        assert all(v==0 for v in json.loads((reference_path.parent/'lifecycle.json').read_text()).values())
    attention=candidate['long_npu_attention_counts']
    if candidate['variant'].startswith('full-batch'):
        assert attention['min_b1_length_exclusive']==2048 and attention['optimized']==0
        assert attention['bypass']==18*len(candidate['records'])
        assert all(r['long_npu_attention_handles']==0 for r in candidate['records'])
    else:
        assert attention['min_b1_length_exclusive']==615 and attention['optimized']==108*len(candidate['records'])
        assert all(r['long_npu_attention_handles']==4 for r in candidate['records'])
    assert attention['fallback']==0 and not attention['errors']
    assert candidate['counts_after']['projections']['runtime_fallback']==candidate['counts_before']['projections']['runtime_fallback']
    assert not candidate['counts_after']['projections']['errors']
    assert not candidate['long_mlp_counts']['runtime_errors']
    assert candidate['long_mlp_counts']['range_fallbacks']==0
    precision=candidate.get('medium_precision_counts',{})
    if precision:assert precision['batch_shapes']=={'8x1181':1,'8x1193':1,'2x1105':1} and precision['scopes']==3
    sparse=candidate.get('medium_sparse_counts',{})
    if sparse:assert sparse['read_errors']==0 and sparse['repair']['density_rejections']==0
    flash=candidate.get('medium_flash_counts',{})
    if flash:assert flash['fallback']==0 and not flash['errors'] and flash['optimized']==18*len(candidate['records'])
    dense=candidate.get('medium_dense_counts',{})
    if dense:assert dense['linear_calls']>0 and not dense['errors']
    manifest=json.loads((candidate_path.parent.parent/'coverage-manifest.json').read_text())
    sample_path=Path('output/decision-index-audit/sample-1000.jsonl.gz')
    assert sha256_file(sample_path,gunzip=True)==manifest['source_sample_rows_sha256']
    meta=next(r for r in manifest['cases'] if r['id']==candidate['case'])
    sample=list(read_jsonl(sample_path));row=next(r for r in sample if r['_evaluation']['run_id']==meta['run_id'])
    group=[r for r in sample if (r['_evaluation']['catalog_id'],r['_evaluation']['group_id'])==(row['_evaluation']['catalog_id'],row['_evaluation']['group_id'])]
    assert len(group)==1 and row['_evaluation']['catalog_id']==9,'Requires complete single-request Home-Appliance group'
    def native(response,seconds):
        report=score(group,{meta['run_id']:{'status':'ok','response':response,'total_wall_ms':seconds*1000}})
        return {k:v for k,v in report.items() if 'latency' not in k}
    expected=reference['response'];cpu_score=native(expected,reference['seconds'])
    parent_score=native(parent['response'],parent['seconds'])
    parent_parity=compare(parent['response'],expected,False)
    parent_qualified=(parent_parity['max_delta']<=.01 and all(parent_parity[k] for k in ('choices_unchanged','usage_equal','model_equal')))
    # A numerically rejected parent's accidental extra correct prediction is
    # disclosed, but cannot redefine the canonical model's expected decision.
    baselines=(cpu_score,parent_score) if parent_qualified else (cpu_score,)
    rows=[]
    for record in candidate['records']:
        response=record['response'];parity=compare(response,expected,False)
        assert set(response['answers'])==set(row['questions']) and not any('error' in v for v in response['answers'].values())
        task=native(response,record['seconds'])
        passed=(parity['max_delta']<=.01 and all(parity[k] for k in ('choices_unchanged','usage_equal','model_equal'))
                and all(task[metric]>=baseline[metric] for metric in ('field_accuracy','case_exact_accuracy')
                        for baseline in baselines))
        rows.append({'seconds':record['seconds'],'parity_without_confidence':parity,
                     'all_numeric_max_delta':compare(response,expected)['max_delta'],'native_metrics':task,'passed':passed})
    lifecycle=json.loads((candidate_path.parent/'lifecycle.json').read_text());assert all(v==0 for v in lifecycle.values())
    result={'passed':all(r['passed'] for r in rows),'profile':candidate['profile'],'scope':'One complete18-question development case, not full Home-Appliance or full-suite accuracy',
            'case':candidate['case'],'cpu_reference_metrics':cpu_score,'parent_vs_cpu':parent_parity,
            'parent_qualified_as_numerical_baseline':parent_qualified,
            'parent_metrics':parent_score,
            'records':rows,'median_seconds':statistics.median(r['seconds'] for r in rows),
            'parent_median_seconds':statistics.median(r['seconds'] for r in parent['records']),
            'after_close':lifecycle,'cpu_reference_used_only_for_correctness':True,
            'reference_kind':reference['reference_kind'],
            'reference_sha256':hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    target=Path(output) if output else candidate_path.parent/'home-appliance-check.json'
    if target.exists():raise FileExistsError('Preserve prior check evidence: '+str(target))
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('candidate');p.add_argument('cpu_reference');p.add_argument('parent');p.add_argument('--output',type=Path);a=p.parse_args()
    assert check(*map(Path,(a.candidate,a.cpu_reference,a.parent)),output=a.output)['passed'],'Development case failed numerical/decision/task gate'
