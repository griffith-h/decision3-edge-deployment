"""Three complete AB/BA request pairs; cold bounded cache on every new call."""
import hashlib,json,statistics,time

PAYLOAD_SHA='acecf821f479ac8a7af688d9bcfce7affd4bf578c72a3e70f0271cce361603bb'
EXPECTED={'mkldnn':True,'threads':3,'float32_precision':'highest','autocast_cpu':False}

def run(model,owner,request,backend,operating_state,memory_snapshot,emit,out):
    payload_sha=hashlib.sha256(json.dumps(request,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    if payload_sha!=PAYLOAD_SHA:raise ValueError('Fixed B2 request changed')
    start=time.perf_counter_ns()
    from original_layout_scope import OriginalLayoutScope
    scope=OriginalLayoutScope(owner);install_seconds=(time.perf_counter_ns()-start)/1e9
    records=[];warmups=[]
    def counts():
        state=owner.medium_sparse
        return {'projection_calls':owner.engine.calls,'native_mlp_calls':owner.native_mlp.calls,
                'attention_calls':owner.npu_calls,'graph_calls':owner.bank.counts.get('graph_calls',0),
                **{'repair_'+k:state.repair.counts[k] for k in ('calls','selected','changed','density_rejections')},
                'read_errors':state.counts['read_errors'],'cpu_fallbacks':owner.engine.cpu_fallbacks}
    def empty():
        assert not scope.kernel.active and scope.kernel.live_bytes==0 and not scope.kernel.cache,'Request cache leaked'
    def save(completed=False):
        data={'completed':completed,'payload_sha256':payload_sha,'pair_count':3,
              'scope':'One fixed B2/T1193 full model request, not HTTP, full18 or official accuracy',
              'protocol':'old/new,new/old,old/new after one full shape warmup per mode; candidate cache cold each request',
              'cost_rule':'Full system_one wall includes copies/evictions/releases; conservatively subtract whole one-off scope installation from each pair saving',
              'memory_rule':'Exact live padded payload <=64MiB; cumulative allocation is traffic, not live memory. cgroup/RSS include model/SDK/allocator; cumulative peak cannot identify a per-mode increment or be added to RSS.',
              'install_seconds':install_seconds,'warmups':warmups,'records':records,
              'cache_limit_bytes':scope.kernel.max_bytes,'cache_counts':dict(scope.kernel.counts),
              'final_memory':memory_snapshot()}
        if completed:
            pairs=[]
            for i in range(3):
                pair={r['mode']:r for r in records if r['pair']==i};old=pair['old']['seconds'];new=pair['new']['seconds']
                pairs.append({'pair':i,'old_seconds':old,'new_seconds':new,'new_over_old':new/old,
                              'gross_saved_seconds':old-new,'net_saved_seconds':old-new-install_seconds})
            data['pairs']=pairs;data['median_net_saved_seconds']=statistics.median(p['net_saved_seconds'] for p in pairs)
            data['median_seconds']={m:statistics.median(r['seconds'] for r in records if r['mode']==m) for m in ('old','new')}
            data['speed_gate_passed']=data['median_net_saved_seconds']>1.0
        (out/'original-layout-pairs.json').write_text(json.dumps(data,indent=2)+'\n')
    def call(mode,pair=None):
        scope.use(mode=='new');empty()
        before=counts();cache_before=dict(scope.kernel.counts);bb=backend('before_'+mode)
        memory_before=memory_snapshot();condition=operating_state()
        assert all(bb[k]==v for k,v in EXPECTED.items())
        emit({'event':'request_begin','phase':'shape_warmup' if pair is None else 'paired_request','mode':mode,'pair':pair,'budget_seconds':120})
        cpu_start=time.process_time_ns();start=time.perf_counter_ns()
        response=model.system_one(**request);elapsed=time.perf_counter_ns()-start;cpu_elapsed=time.process_time_ns()-cpu_start
        empty();ba=backend('after_'+mode);after=counts()
        assert all(ba[k]==v for k,v in EXPECTED.items())
        delta={k:after[k]-before[k] for k in before}
        assert not any(delta[k] for k in ('repair_density_rejections','read_errors','cpu_fallbacks'))
        cache_delta={k:scope.kernel.counts[k]-cache_before[k] for k in cache_before if k!='peak_bytes'}
        if mode=='new':
            assert cache_delta['requests']==1 and cache_delta['calls']==delta['repair_calls'] and cache_delta['selected']==delta['repair_selected']
            assert not any(cache_delta[k] for k in ('invalidations','oversize_bypasses'))
            assert 0<scope.kernel.counts['peak_bytes']<=scope.kernel.max_bytes
        else:assert not any(cache_delta.values())
        return {'mode':mode,'pair':pair,'seconds':elapsed/1e9,'process_cpu_seconds':cpu_elapsed/1e9,
                'response':response,'counter_delta':delta,'cache_delta':cache_delta,
                'cache_peak_payload_bytes':scope.kernel.counts['peak_bytes'] if mode=='new' else 0,
                'cache_empty_after':True,'backend_before':bb,'backend_after':ba,
                'memory_before':memory_before,'memory_after':memory_snapshot(),
                'operating_state_before':condition,'operating_state_after':operating_state()}
    try:
        emit({'event':'ready','case':'original-layout-pairs','install_seconds':install_seconds})
        for mode in ('old','new'):
            record=call(mode);warmups.append(record);save()
            assert record['response']==warmups[0]['response'],'Warmup changed full response'
            assert record['counter_delta']==warmups[0]['counter_delta'],'Warmup changed call/selection counters'
            emit({'event':'case_complete','phase':'shape_warmup','mode':mode,'seconds':record['seconds']})
        for pair in range(3):
            for mode in (('old','new') if pair%2==0 else ('new','old')):
                record=call(mode,pair);records.append(record);save()
                assert record['response']==warmups[0]['response'],'Layout changed full response'
                assert record['counter_delta']==warmups[0]['counter_delta'],'Layout changed call/selection counters'
                emit({'event':'case_complete','pair':pair,'mode':mode,'seconds':record['seconds'],
                      'prepare_seconds':record['cache_delta']['prepare_seconds'],'release_seconds':record['cache_delta']['release_seconds']})
        save(True);emit({'event':'complete','pairs':3,'response_equal':True})
    finally:
        scope.close()
        (out/'original-layout-restored.json').write_text(json.dumps({'kernel_restored':owner.medium_sparse.repair.blas_repair is scope.old_kernel,
            'request_restored':model.system_one is scope.old_request,'cache_empty':not scope.kernel.cache and not scope.kernel.active and scope.kernel.live_bytes==0,'closed':scope.closed})+'\n')
