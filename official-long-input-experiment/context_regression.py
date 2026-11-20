"""Full frozen regression with explicit request-entry/exit backend identity.

References use independent owners and the same public-request state boundary.
No gold outputs are mounted on the inference board; thresholds live offline.
"""
import copy,hashlib,json,time

EXPECTED={'mkldnn':True,'threads':3,'float32_precision':'highest','autocast_cpu':False}

def require_backend(state):
    if any(state.get(k)!=v for k,v in EXPECTED.items()):
        raise RuntimeError('Regression backend context changed: '+repr(state))

def run(model,owner,rows,get_counts,backend,emit,out,variant,input_bytes,counters):
    if len(rows)!=34 or len({r['id'] for r in rows})!=34:raise ValueError('Requires all34 unique frozen cases')
    records=[];data={'completed':False,'variant':variant,'context':EXPECTED,
        'input_sha256':hashlib.sha256(input_bytes).hexdigest(),'case_ids':[r['id'] for r in rows],
        'reference_kind':'Independent owner, unchanged public requests and original batches; request-entry/exit context fixed; internal image fallback preserved',
        'records':records}
    target=out/'context-regression.json'
    def save():target.write_text(json.dumps(data)+'\n')
    save();emit({'event':'ready','case':'context-regression34','variant':variant})
    for row in rows:
        before=backend('before_'+row['id']);require_backend(before)
        counts_before=copy.deepcopy(get_counts());matrix_before=counters(owner)
        emit({'event':'request_begin','case':row['id']});start=time.perf_counter()
        try:response=model.system_one(**row['payload'])
        except Exception as exc:
            data['failure']={'id':row['id'],'error':type(exc).__name__+': '+str(exc),'backend_after':backend('failure_'+row['id'])};save();raise
        elapsed=time.perf_counter()-start;after=backend('after_'+row['id'])
        record={'id':row['id'],'response':response,'seconds':elapsed,'backend_before':before,'backend_after':after,
            'counts_before':counts_before,'counts_after':copy.deepcopy(get_counts()),
            'npu_calls_delta':[a-b for a,b in zip(counters(owner),matrix_before)],
            'request_scope':dict(owner.request_backend_scope_counts)}
        records.append(record);save();require_backend(after)
        if variant=='reference' and any(record['npu_calls_delta']):raise RuntimeError('Original CPU reference unexpectedly used NPU/graph')
        emit({'event':'case_complete','case':row['id'],'seconds':elapsed})
    data['completed']=True;save();emit({'event':'complete','cases':len(records)})
