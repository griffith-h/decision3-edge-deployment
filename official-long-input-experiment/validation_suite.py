"""Serial development validation, including long-to-short bank transitions."""
import copy,json,resource,time
from pathlib import Path

ORDER=('multi4605','protect615','protect115','multi4597','tools32','short314','medium997','medium1659')

def run(model,owner,cases,get_counts,baseline_logits,emit,out):
    from board import operating_state
    records=[]
    for name in ORDER:
        before=copy.deepcopy(get_counts());state=operating_state()
        emit({'event':'request_begin','case':name});start=time.perf_counter()
        response=model.system_one(**cases[name]);seconds=time.perf_counter()-start
        record={'id':name,'seconds':seconds,'response':response,'counts_before':before,
                'counts_after':get_counts(),'engine_handles':len(owner.engine.handles),
                'native_mlp_handles':len(owner.native_mlp.handles),'encoder_restored':model.encoder is owner.last_readout.encoder,'operating_state':state,
                'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        records.append(record)
        (out/'validation.json').write_text(json.dumps({'completed':False,'records':records})+'\n')
        if name=='tools32':
            (out/'tools-result.json').write_text(json.dumps({'completed':True,'response':response,
                'seconds':seconds,'records':[record]})+'\n')
        emit({'event':'case_complete','case':name,'seconds':seconds})

    regression=[]
    for row in json.loads(Path('/inputs/regression-inputs.json').read_text()):
        emit({'event':'request_begin','case':row['id']});start=time.perf_counter()
        response=model.system_one(**row['payload']);seconds=time.perf_counter()-start
        regression.append({'id':row['id'],'seconds':seconds,'response':response})
        (out/'regression.json').write_text(json.dumps({'completed':False,'records':regression})+'\n')
        emit({'event':'case_complete','case':row['id'],'seconds':seconds})
    (out/'regression.json').write_text(json.dumps({'completed':True,'records':regression})+'\n')

    # Match the existing five warm pairs per shape and AB/BA order exactly.
    paired=[];candidate_logits=model.logits
    for name in ('protect115','protect615'):
        model.system_one(**cases[name])
        for repeat in range(5):
            for enabled in ((False,True) if repeat%2==0 else (True,False)):
                state=operating_state();saved=model.logits
                model.logits=candidate_logits if enabled else baseline_logits
                emit({'event':'request_begin','case':name});start=time.perf_counter()
                try:response=model.system_one(**cases[name]);seconds=time.perf_counter()-start
                finally:model.logits=saved
                paired.append({'case':name,'candidate':enabled,'seconds':seconds,'response':response,'operating_state':state})
                (out/'protection.json').write_text(json.dumps({'completed':False,'records':paired})+'\n')
    (out/'protection.json').write_text(json.dumps({'completed':True,'records':paired})+'\n')
    (out/'validation.json').write_text(json.dumps({'completed':True,'records':records})+'\n')
    emit({'event':'complete','development_calls':len(records),'regression_cases':len(regression),'paired_calls':len(paired)})
