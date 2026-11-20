"""Instrumented development coverage; stops on a new slow/failed request."""
import copy,hashlib,json,math,resource,time
from pathlib import Path
from types import MethodType

def response_issues(response,questions):
    issues=[]
    answers=response.get('answers')
    if not isinstance(answers,dict) or set(answers)!=set(questions):issues.append('answer_fields')
    if isinstance(answers,dict) and any(not isinstance(v,dict) or 'error' in v for v in answers.values()):issues.append('field_error')
    def finite(value):
        if isinstance(value,float):return math.isfinite(value)
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,list):return all(finite(v) for v in value)
        return True
    if not finite(response):issues.append('nonfinite_output')
    return issues

def run(model,owner,get_counts,emit,out):
    path=Path('/inputs/coverage-inputs.json');manifest=json.loads(Path('/inputs/coverage-manifest.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['input_file_sha256']
    rows=json.loads(path.read_text());metadata={r['id']:r for r in manifest['cases']}
    assert [r['id'] for r in rows]==list(metadata)
    cursor_path=Path('/inputs/coverage-start.json')
    start=json.loads(cursor_path.read_text())['start_index'] if cursor_path.exists() else 0
    assert type(start) is int and 0<=start<len(rows)
    summary={'completed':False,'instrumented':True,'scope':'development coverage view; not final full1000 performance',
        'input_sha256':manifest['input_file_sha256'],'source_rows_sha256':manifest['source_sample_rows_sha256'],
        'start_index':start,'attempted':0,'completed_requests':0,'total_requests':len(rows)}
    (out/'coverage-status.json').write_text(json.dumps(summary)+'\n')
    with (out/'coverage-results.jsonl').open('w') as results:
        for index in range(start,len(rows)):
            row=rows[index];meta=metadata[row['id']];batches=[];original=model.logits
            active={'id':row['id'],'index':index,'run_id':meta['run_id'],'batches':batches,'completed':False}
            (out/'active-case.json').write_text(json.dumps(active)+'\n')
            def observe(instance,sequences,counts):
                lengths=[len(s) for s in sequences]
                batches.append({'lengths':lengths,'padded_tokens':len(lengths)*max(lengths),'effective_tokens':sum(lengths)})
                (out/'active-case.json').write_text(json.dumps(active)+'\n')
                return original(sequences,counts)
            model.logits=MethodType(observe,model);before=copy.deepcopy(get_counts())
            emit({'event':'request_begin','case':row['id'],'index':index})
            start_time=time.perf_counter()
            try:response=model.system_one(**row['payload']);seconds=time.perf_counter()-start_time
            finally:model.logits=original
            issues=response_issues(response,row['payload']['questions'])
            if model.encoder is not owner.last_readout.encoder:issues.append('encoder_not_restored')
            errors=bool(issues)
            record={'id':row['id'],'index':index,'run_id':meta['run_id'],'seconds':seconds,'response':response,
                'counts_before':before,'counts_after':get_counts(),'batches':batches,'issues':issues,
                'encoder_restored':model.encoder is owner.last_readout.encoder,
                'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            results.write(json.dumps(record)+'\n');results.flush()
            active.update(completed=True,seconds=seconds);(out/'active-case.json').write_text(json.dumps(active)+'\n')
            summary['attempted']+=1;summary['completed_requests']+=1;summary['next_index']=index+1
            summary['completed']=index+1==len(rows)
            if seconds>30 or errors:summary.update(completed=False,stop_reason='slow' if seconds>30 else 'field_error',stop_id=row['id'])
            (out/'coverage-status.json').write_text(json.dumps(summary)+'\n')
            emit({'event':'case_complete','case':row['id'],'index':index,'seconds':seconds,'field_errors':errors})
            if seconds>30 or errors:break
    emit({'event':'complete',**summary})
