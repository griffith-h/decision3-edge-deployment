"""Same-owner AB/BA request timing, switching only the FP32 copy function."""
import ctypes,hashlib,json,statistics,time
from pathlib import Path

def run(model,owner,request,emit,out):
    state=owner.medium_sparse;root=Path(__file__).parent
    baseline=json.loads((root/'native-reader-baseline.json').read_text())
    current=json.loads((root/'linear-precision-build.json').read_text())
    if baseline['build']['sha256']['native_split_reference.inc']!=current['sha256']['native_split_reference.inc']:
        raise ValueError('Owned native Matrix ABI source changed')
    path=root/'liblinear_precision_before_layout.so'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=baseline['library_sha256']:
        raise ValueError('Previous native reader identity mismatch')
    previous_lib=ctypes.CDLL(str(path));previous=previous_lib.d3_probe_copy_fp32
    previous.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p];previous.restype=ctypes.c_int
    blocked=state.raw_native.d3_probe_copy_fp32
    records=[];warm=None
    def counts():
        return {'projection_calls':owner.engine.calls,'native_mlp_calls':owner.native_mlp.calls,
                'attention_calls':owner.npu_calls,'graph_calls':owner.bank.counts.get('graph_calls',0),
                'repair_calls':state.repair.counts['calls'],'selected':state.repair.counts['selected'],
                'changed':state.repair.counts['changed'],'density_rejections':state.repair.counts['density_rejections'],
                'read_errors':state.counts['read_errors'],'cpu_fallbacks':owner.engine.cpu_fallbacks}
    def call(mode,phase,pair=None):
        if state.group_active:raise RuntimeError('Cannot switch reader during projection workers')
        state.raw_native.d3_probe_copy_fp32=previous if mode=='previous' else blocked
        before=counts();emit({'event':'request_begin','phase':phase,'mode':mode,'pair':pair,'budget_seconds':120})
        start=time.perf_counter();response=model.system_one(**request);seconds=time.perf_counter()-start
        after=counts();delta={k:after[k]-before[k] for k in before}
        if any(delta[k] for k in ('density_rejections','read_errors','cpu_fallbacks')):
            raise AssertionError('Copy comparison triggered a runtime fallback/error')
        return {'mode':mode,'pair':pair,'seconds':seconds,'response':response,'counter_delta':delta}
    def save(completed=False):
        data={'completed':completed,'scope':'Same B2/T1193 request, same owner, reader function only; not full-suite performance',
              'warmup':warm,'records':records,'pair_count':3,
              'warmup_is_preserved_but_not_a_paired_measurement':True}
        if completed:
            ratios=[]
            for index in range(3):
                pair={r['mode']:r for r in records if r['pair']==index}
                ratios.append(pair['blocked']['seconds']/pair['previous']['seconds'])
            data['paired_ratios']=ratios;data['median_pair_ratio']=statistics.median(ratios)
            data['median_seconds']={mode:statistics.median(r['seconds'] for r in records if r['mode']==mode)
                                    for mode in ('previous','blocked')}
        (out/'native-copy-pairing.json').write_text(json.dumps(data,indent=2)+'\n')
    try:
        emit({'event':'ready','case':'copy-layout-pairing','variant':'same_owner_reader_only'})
        warm=call('blocked','preserved_shape_warmup');save()
        for index in range(3):
            for mode in (('previous','blocked') if index%2==0 else ('blocked','previous')):
                record=call(mode,'paired_request',index)
                if record['response']!=warm['response']:raise AssertionError('Reader change altered complete response')
                if records and record['counter_delta']!=records[0]['counter_delta']:
                    raise AssertionError('Reader change altered inference call/selection counts')
                records.append(record);save()
                emit({'event':'case_complete','mode':mode,'pair':index,'seconds':record['seconds']})
        save(True);emit({'event':'complete','pairs':3,'response_bitwise_equal':True})
    finally:state.raw_native.d3_probe_copy_fp32=blocked
