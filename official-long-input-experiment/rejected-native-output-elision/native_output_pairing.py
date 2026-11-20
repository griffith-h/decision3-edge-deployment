"""Same-owner AB/BA request timing, retaining all math and native guards."""
import ctypes,hashlib,json,statistics,time
from pathlib import Path

def run(model,owner,request,emit,out):
    state=owner.medium_sparse;root=Path(__file__).parent
    checked=state.native_compute_no_output
    records=[];warm=None
    def counts():
        return {'projection_calls':owner.engine.calls,'native_mlp_calls':owner.native_mlp.calls,
                'attention_calls':owner.npu_calls,'graph_calls':owner.bank.counts.get('graph_calls',0),
                'repair_calls':state.repair.counts['calls'],'selected':state.repair.counts['selected'],
                'changed':state.repair.counts['changed'],'density_rejections':state.repair.counts['density_rejections'],
                'read_errors':state.counts['read_errors'],'cpu_fallbacks':owner.engine.cpu_fallbacks}
    def call(mode,phase,pair=None):
        if state.group_active or owner.long_mlp.active:raise RuntimeError('Cannot switch output policy during inference')
        state.native_compute_no_output=None if mode=='copy' else checked
        elided_before=state.counts['elided_native_outputs']
        before=counts();emit({'event':'request_begin','phase':phase,'mode':mode,'pair':pair,'budget_seconds':120})
        start=time.perf_counter();response=model.system_one(**request);seconds=time.perf_counter()-start
        after=counts();delta={k:after[k]-before[k] for k in before}
        if any(delta[k] for k in ('density_rejections','read_errors','cpu_fallbacks')):
            raise AssertionError('Copy comparison triggered a runtime fallback/error')
        elided=state.counts['elided_native_outputs']-elided_before
        if (mode=='copy' and elided) or (mode=='no_output' and elided<=0):raise AssertionError('Output elision path not exercised as requested')
        return {'mode':mode,'pair':pair,'seconds':seconds,'response':response,'counter_delta':delta,'elided_outputs':elided}
    def save(completed=False):
        data={'completed':completed,'scope':'Same B2/T1193 request, same owner; only discarded BF16 outputs omitted with the old guard; not full-suite performance',
              'warmup':warm,'records':records,'pair_count':3,
              'warmup_is_preserved_but_not_a_paired_measurement':True}
        if completed:
            ratios=[]
            for index in range(3):
                pair={r['mode']:r for r in records if r['pair']==index}
                ratios.append(pair['no_output']['seconds']/pair['copy']['seconds'])
            data['paired_ratios']=ratios;data['median_pair_ratio']=statistics.median(ratios)
            data['median_seconds']={mode:statistics.median(r['seconds'] for r in records if r['mode']==mode)
                                    for mode in ('copy','no_output')}
        (out/'native-output-pairing.json').write_text(json.dumps(data,indent=2)+'\n')
    try:
        emit({'event':'ready','case':'native-output-pairing','variant':'same_owner_discarded_output_only'})
        warm=call('no_output','preserved_shape_warmup');save()
        for index in range(3):
            for mode in (('copy','no_output') if index%2==0 else ('no_output','copy')):
                record=call(mode,'paired_request',index)
                records.append(record);save()
                if record['response']!=warm['response']:raise AssertionError('Output policy altered complete response')
                if records and record['counter_delta']!=records[0]['counter_delta']:
                    raise AssertionError('Output policy altered inference call/selection counts')
                emit({'event':'case_complete','mode':mode,'pair':index,'seconds':record['seconds']})
        save(True);emit({'event':'complete','pairs':3,'response_bitwise_equal':True})
    finally:state.native_compute_no_output=checked
