"""Audit complete cold-cache request pairs against frozen qualified/CPU outputs."""
import argparse,hashlib,json,statistics
from pathlib import Path
from summarize import compare

ROOT=Path(__file__).resolve().parent.parent
ITER=ROOT/'output/decision-index-audit/iterations'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def review(folder):
    execution=load(folder/'execution.json');data=load(folder/'original-layout-pairs.json')
    assert execution['iteration']=='r093' and execution['case']=='original-layout-pairs' and not execution['profile']
    assert execution['request_completed'] and execution['exit_code']==0 and execution['retained_healthy'] and not execution['cleanup_errors']
    assert not execution.get('budget_exceeded') and not execution['worker_final_state']['OOMKilled']
    assert all(load(folder/'original-layout-restored.json').values())
    assert all(v==0 for v in load(folder/'lifecycle.json').values())
    checks=load(folder/'isolated-boundary-checks.json')
    assert checks['passed'] and checks['tests_run']==81 and checks['skipped']==0
    assert data['completed'] and data['pair_count']==3
    assert data['payload_sha256']=='acecf821f479ac8a7af688d9bcfce7affd4bf578c72a3e70f0271cce361603bb'
    assert data['cache_limit_bytes']==64*1024*1024
    expected={'mkldnn':True,'threads':3,'float32_precision':'highest','autocast_cpu':False}
    before_cleanup=load(folder/'runtime-backend-before-cleanup.json')
    assert all(before_cleanup[k]==v for k,v in expected.items())
    pins=load(folder.parent/'r093-source-pin.json')['sha256'];sources=load(folder/'lab-sources.json')
    assert all(sources[k]==v for k,v in pins.items())
    assert sources['libsparse_cpu.so']=='17da1be76215bdf3ffd58548205a44c453d3a4b911655254c22b896237e216a0'
    inherited=load(ITER/'r092/20261020T134215-selected-layout-original4-preflight-full-batch-sparse-request-scope-plain/lab-sources.json')
    assert all(sources[k]==v for k,v in inherited.items() if k!='probe.py'),'Inherited source/binary changed'
    qpath=ITER/'r072/20261020T004105-diagnostic-home-pair-full-batch-sparse-band25-plain/result.json'
    cpath=ITER/'r061/20261019T170841-diagnostic-home-pair-cpu-full-plain/result.json'
    qualified=load(qpath)['response'];cpu=load(cpath)['response']
    records=data['records'];warmups=data['warmups']
    assert [(r['pair'],r['mode']) for r in records]==[(0,'old'),(0,'new'),(1,'new'),(1,'old'),(2,'old'),(2,'new')]
    assert [r['mode'] for r in warmups]==['old','new']
    numeric=[]
    for r in [*warmups,*records]:
        assert r['response']==qualified and r['counter_delta']==warmups[0]['counter_delta']
        c=compare(r['response'],cpu,False)
        assert c['max_delta']<=.01 and all(c[k] for k in ('choices_unchanged','noul_predictions_unchanged','usage_equal','model_equal'))
        numeric.append(c)
        assert r['cache_empty_after'] and all(r[b][k]==v for b in ('backend_before','backend_after') for k,v in expected.items())
        cd=r['counter_delta'];cache=r['cache_delta']
        assert cd['repair_calls']==372 and cd['repair_selected']==8063030
        assert not any(cd[k] for k in ('repair_density_rejections','read_errors','cpu_fallbacks'))
        if r['mode']=='new':
            assert cache['builds']==72 and cache['hits']==300 and cache['evictions']==62
            assert cache['calls']==372 and cache['selected']==8063030 and cache['requests']==1
            assert not cache['invalidations'] and not cache['oversize_bypasses']
            assert cache['allocated_bytes']==490340352 and r['cache_peak_payload_bytes']==66748416
        else:assert not any(cache.values()) and r['cache_peak_payload_bytes']==0
        for name in ('memory_before','memory_after'):
            mem=r[name];assert int(mem['memory.peak'])<=int(mem['memory.max'])
            events=dict(line.split() for line in mem['memory.events'].splitlines())
            assert not any(int(events[k]) for k in ('max','oom','oom_kill'))
    pairs=[]
    for i in range(3):
        pair={r['mode']:r for r in records if r['pair']==i};old=pair['old']['seconds'];new=pair['new']['seconds']
        pairs.append({'pair':i,'old_seconds':old,'new_seconds':new,'new_over_old':new/old,
                      'gross_saved_seconds':old-new,'net_saved_seconds':old-new-data['install_seconds']})
    assert pairs==data['pairs']
    net=statistics.median(p['net_saved_seconds'] for p in pairs)
    assert data['median_net_saved_seconds']==net and data['speed_gate_passed']==(net>1.0)
    memory_peak=max(int(r[m]['memory.peak']) for r in [*warmups,*records] for m in ('memory_before','memory_after'))
    report={'iteration':'r093','passed_numerical_and_runtime':True,'speed_gate_passed':net>1.0,
            'scope':data['scope'],'payload_sha256':data['payload_sha256'],'arm_checks':checks['tests_run'],
            'pairs':pairs,'median_net_saved_seconds':net,'median_seconds':data['median_seconds'],
            'install_seconds':data['install_seconds'],'cost_rule':data['cost_rule'],'memory_rule':data['memory_rule'],
            'full_response_equal_qualified':True,'maximum_cpu_nonconfidence':max(r['max_delta'] for r in numeric),
            'choices_noul_usage_model_equal':True,'cache_limit_bytes':data['cache_limit_bytes'],
            'cache_counts':data['cache_counts'],'whole_run_cgroup_peak_bytes':memory_peak,
            'warmups_and_records':[{k:v for k,v in r.items() if k not in ('response','operating_state_before','operating_state_after')} for r in [*warmups,*records]],
            'service_healthy':True,'resources_closed':True,'cache_and_methods_restored':True,'no_oom':True,
            'sources_sha256':sha(folder/'lab-sources.json'),'record_sha256':sha(folder/'original-layout-pairs.json'),
            'qualified_reference_sha256':sha(qpath),'cpu_reference_sha256':sha(cpath),
            'gate_source_sha256':sha(Path(__file__).with_name('check_regression.py')),
            'not_full34_or_full18':True,'not_deployed':True}
    (folder/'original-layout-pairs-review.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=review(a.folder)
    print(json.dumps({k:v for k,v in r.items() if k not in ('warmups_and_records','cache_counts')},indent=2))
