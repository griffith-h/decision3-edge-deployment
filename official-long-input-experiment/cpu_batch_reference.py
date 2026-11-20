"""Canonical current-CPU reference at the unchanged original8-row boundaries.

Transport is split only at original model batches. B/T, sequence order, masks,
and logits calls remain identical; this is not a speed benchmark.
"""
import json,time

def counters(owner):
    return (owner.engine.calls,owner.native_mlp.calls,owner.npu_calls,
            owner.bank.counts.get('graph_calls',0))

def run(model,owner,request,emit,out,case):
    assert model.batch_size==8 and model.max_length is None
    prepared=model.prepare(**request);keys=list(prepared.runnable)
    assert set(keys)==set(request['questions']) and not prepared.errors,'Reference requires all original questions runnable'
    records=[];answers={};tokens=0;seconds=0;start_counts=counters(owner)
    def save(completed=False):
        response={'model':model.model_name,'answers':answers,'usage':{'input_tokens':tokens,'output_tokens':0}}
        data={'completed':completed,'case':case,'variant':'cpu-full','profile':False,'response':response,
              'seconds':seconds,'records':records,'npu_matrix_calls':counters(owner)[0]-start_counts[0],
              'reference_kind':'Current CPU, original batch_size8 grouping/order/padding, no row-wise split; correctness only'}
        (out/'result.json').write_text(json.dumps(data)+'\n')
    for index,start in enumerate(range(0,len(keys),model.batch_size)):
        group=keys[start:start+model.batch_size]
        item=dict(request,questions={key:request['questions'][key] for key in group})
        subset=model.prepare(**item)
        assert list(subset.runnable)==group and all(subset.sequences[k]==prepared.sequences[k] for k in group)
        lengths=[len(prepared.sequences[k]) for k in group];before=counters(owner)
        emit({'event':'request_begin','cpu_reference_batch':index,'batch_size':len(group),'width':max(lengths)})
        begin=time.perf_counter();response=model.system_one(**item);elapsed=time.perf_counter()-begin
        row={'batch_index':index,'lengths':lengths,'width':max(lengths),'seconds':elapsed,
             'response':response,'npu_calls_delta':[a-b for a,b in zip(counters(owner),before)]}
        records.append(row);seconds+=elapsed;answers.update(response['answers']);tokens+=response['usage']['input_tokens'];save()
        assert before==counters(owner),'CPU batch reference unexpectedly used NPU or graph bank'
        assert set(response['answers'])==set(group) and not any('error' in a for a in response['answers'].values())
        assert response['usage']['input_tokens']==sum(lengths) and response['usage']['output_tokens']==0
        emit({'event':'case_complete','cpu_reference_batch':index,'seconds':elapsed})
    assert set(answers)==set(request['questions']) and counters(owner)==start_counts
    save(True);emit({'event':'complete','reference_batches':len(records),'questions':len(answers)})
