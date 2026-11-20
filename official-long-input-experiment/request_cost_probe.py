"""Three interleaved plain/ledger pairs on the fixed qualified B2/T1193 request."""
import copy,ctypes,hashlib,json,os,statistics,time
from pathlib import Path
from request_cost_ledger import install,partition

class SdkEvent(ctypes.Structure):
    _fields_=[(name,ctypes.c_uint64) for name in ('start_ns','end_ns','cpu_start_ns','cpu_end_ns','thread','context')]+[('status',ctypes.c_int64)]

def run(model,owner,request,get_counts,backend,operating_state,emit,out):
    root_path=Path(__file__).parent;build=json.loads((root_path/'request-sdk-trace-build.json').read_text())
    for name,digest in build['sha256'].items():
        assert hashlib.sha256((root_path/name).read_bytes()).hexdigest()==digest,'SDK diagnostic build mismatch'
    lib=ctypes.CDLL(None)
    lib.ledger_sdk_begin.restype=None;lib.ledger_sdk_end.restype=ctypes.c_uint64
    lib.ledger_sdk_events.restype=ctypes.POINTER(SdkEvent);lib.ledger_sdk_capacity.restype=ctypes.c_uint64
    records=[];warm=None;rec=None
    payload_hash=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    def counts():
        state=owner.medium_sparse
        return {'projection_calls':owner.engine.calls,'native_mlp_calls':owner.native_mlp.calls,
                'attention_calls':owner.npu_calls,'graph_calls':owner.bank.counts.get('graph_calls',0),
                **{'repair_'+k:state.repair.counts[k] for k in ('calls','selected','changed','density_rejections')},
                'read_errors':state.counts['read_errors'],'cpu_fallbacks':owner.engine.cpu_fallbacks}
    def save(completed=False):
        data={'completed':completed,'scope':'One fixed B2/T1193 model request; profiling diagnosis, not HTTP or suite performance',
              'payload_sha256':payload_hash,'warmup':warm,'records':records,'pair_count':3,
              'sdk_time_means':'Host synchronous public SDK entry to return; includes driver submission/wait, not pure device execution',
              'ledger_rule':'Owner-thread leaf intervals partition wall time. Concurrent SDK union is overlaid only on native-call/join leaves. Worker/CPU totals are separate overlapping views.'}
        if completed:
            ratios=[]
            for i in range(3):
                pair={r['mode']:r for r in records if r['pair']==i};ratios.append(pair['ledger']['seconds']/pair['plain']['seconds'])
            data['paired_instrumentation_ratios']=ratios;data['median_instrumentation_ratio']=statistics.median(ratios)
            data['median_seconds']={m:statistics.median(r['seconds'] for r in records if r['mode']==m) for m in ('plain','ledger')}
        (out/'request-cost.json').write_text(json.dumps(data,indent=2)+'\n')
    def call(mode,pair=None):
        nonlocal rec
        before_counts=counts();before_backend=backend('before_'+mode);condition=operating_state()
        emit({'event':'request_begin','phase':'preserved_shape_warmup' if pair is None else 'paired_request','mode':mode,'pair':pair,'budget_seconds':120})
        sdk=[];ledger=None
        if mode=='ledger':
            rec=install(model,owner);rec.active=True;lib.ledger_sdk_begin()
            try:
                with rec.span('system_one','request_other') as root:
                    before_cpu=time.process_time_ns();response=model.system_one(**request);after_cpu=time.process_time_ns()
            finally:
                size=lib.ledger_sdk_end();rec.active=False
                if size>lib.ledger_sdk_capacity():raise RuntimeError('SDK trace capacity exceeded')
                pointer=lib.ledger_sdk_events()
                sdk=[{name:getattr(pointer[i],name) for name,_ in SdkEvent._fields_} for i in range(size)]
                rec.close()
            elapsed=root['end_ns']-root['start_ns'];ledger=partition(rec.events,root,sdk)
            trace={'events':rec.events,'sdk':sdk,'ledger':ledger,'root':root}
            trace_name='request-cost-trace-'+str(pair)+'.json';(out/trace_name).write_text(json.dumps(trace)+'\n')
            assert sdk and all(s['status']==0 and root['start_ns']<=s['start_ns']<=s['end_ns']<=root['end_ns'] for s in sdk)
            # Detailed trace serialization and hook installation are outside request time.
        else:
            before_cpu=time.process_time_ns();start=time.perf_counter_ns()
            response=model.system_one(**request);elapsed=time.perf_counter_ns()-start;after_cpu=time.process_time_ns();trace_name=None
        after_counts=counts();after_backend=backend('after_'+mode)
        expected={'mkldnn':True,'threads':3,'float32_precision':'highest','autocast_cpu':False}
        assert all(before_backend[k]==after_backend[k]==v for k,v in expected.items())
        delta={k:after_counts[k]-before_counts[k] for k in before_counts}
        assert not any(delta[k] for k in ('repair_density_rejections','read_errors','cpu_fallbacks'))
        result={'mode':mode,'pair':pair,'seconds':elapsed/1e9,'response':response,'counter_delta':delta,
                'backend_before':before_backend,'backend_after':after_backend,'operating_state_before':condition,'operating_state_after':operating_state(),
                'process_cpu_seconds':(after_cpu-before_cpu)/1e9,'trace_file':trace_name,
                'exclusive_seconds':ledger['exclusive_seconds'] if ledger else None,
                'closure_error_ns':ledger['closure_error_ns'] if ledger else None,'sdk_calls':len(sdk) if ledger else None}
        return result
    emit({'event':'ready','case':'request-cost-ledger'})
    try:
        warm=call('plain');save()
        for i in range(3):
            for mode in (('plain','ledger') if i%2==0 else ('ledger','plain')):
                record=call(mode,i);records.append(record);save()
                assert record['response']==warm['response'],'Instrumentation altered complete response'
                assert record['counter_delta']==records[0]['counter_delta'],'Instrumentation changed inference counters'
                emit({'event':'case_complete','mode':mode,'pair':i,'seconds':record['seconds'],'sdk_calls':record['sdk_calls']})
        save(True);emit({'event':'complete','pairs':3,'complete_response_equal':True})
    finally:
        lib.ledger_sdk_end()
        if rec and rec.restores:rec.close()
