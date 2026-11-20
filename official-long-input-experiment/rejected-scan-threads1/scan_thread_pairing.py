"""Same-owner AB/BA request timing, switching only original CPU scan thread scope."""
import ctypes,hashlib,json,statistics,time
from pathlib import Path
import importlib.util
_spec=importlib.util.spec_from_file_location('d3_scan_pair_compare',Path(__file__).with_name('summarize.py'))
_compare_module=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_compare_module)
paired_compare=_compare_module.compare

def run(model,owner,request,emit,out):
    state=owner.medium_sparse;root=Path(__file__).parent
    policy=owner.scan_threads
    records=[];warm=None
    def counts():
        return {'projection_calls':owner.engine.calls,'native_mlp_calls':owner.native_mlp.calls,
                'attention_calls':owner.npu_calls,'graph_calls':owner.bank.counts.get('graph_calls',0),
                'repair_calls':state.repair.counts['calls'],'selected':state.repair.counts['selected'],
                'changed':state.repair.counts['changed'],'density_rejections':state.repair.counts['density_rejections'],
                'read_errors':state.counts['read_errors'],'cpu_fallbacks':owner.engine.cpu_fallbacks}
    def call(mode,phase,pair=None):
        if state.group_active:raise RuntimeError('Cannot switch reader during projection workers')
        policy.enabled=mode=='one'
        scopes_before=policy.counts['scopes']
        before=counts();emit({'event':'request_begin','phase':phase,'mode':mode,'pair':pair,'budget_seconds':120})
        start=time.perf_counter();response=model.system_one(**request);seconds=time.perf_counter()-start
        after=counts();delta={k:after[k]-before[k] for k in before}
        if any(delta[k] for k in ('density_rejections','read_errors','cpu_fallbacks')):
            raise AssertionError('Copy comparison triggered a runtime fallback/error')
        scopes=policy.counts['scopes']-scopes_before
        if scopes!=(6 if mode=='one' else 0) or policy.counts['errors']:raise AssertionError('Unexpected scan scope or errors')
        return {'mode':mode,'pair':pair,'seconds':seconds,'response':response,'counter_delta':delta,'scan_scopes':scopes}
    def save(completed=False):
        data={'completed':completed,'scope':'Same B2/T1193 request, same owner, original CPU scan thread scope only; not full-suite performance',
              'warmup':warm,'records':records,'pair_count':3,
              'warmup_is_preserved_but_not_a_paired_measurement':True}
        if completed:
            ratios=[]
            for index in range(3):
                pair={r['mode']:r for r in records if r['pair']==index}
                ratios.append(pair['one']['seconds']/pair['three']['seconds'])
            data['paired_ratios']=ratios;data['median_pair_ratio']=statistics.median(ratios)
            data['median_seconds']={mode:statistics.median(r['seconds'] for r in records if r['mode']==mode)
                                    for mode in ('three','one')}
        (out/'scan-thread-pairing.json').write_text(json.dumps(data,indent=2)+'\n')
    try:
        emit({'event':'ready','case':'scan-thread-pairing','variant':'same_owner_scan_thread_scope'})
        warm=call('one','preserved_shape_warmup');save()
        for index in range(3):
            for mode in (('three','one') if index%2==0 else ('one','three')):
                record=call(mode,'paired_request',index)
                records.append(record);save()
                parity=paired_compare(record['response'],warm['response'],False)
                if parity['max_delta']>.01 or not all(parity[k] for k in ('choices_unchanged','usage_equal','model_equal')):raise AssertionError('Scan thread change failed paired numerical gate')
                if records and any(record['counter_delta'][k]!=records[0]['counter_delta'][k] for k in ('projection_calls','native_mlp_calls','attention_calls','graph_calls','density_rejections','read_errors','cpu_fallbacks')):
                    raise AssertionError('Reader change altered inference call/selection counts')
                emit({'event':'case_complete','mode':mode,'pair':index,'seconds':record['seconds']})
        save(True);emit({'event':'complete','pairs':3,'paired_numerical_gate_passed':True})
    finally:policy.enabled=True
