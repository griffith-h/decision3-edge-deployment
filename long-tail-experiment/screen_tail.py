from runner import *
from pathlib import Path
app=Owner();old=json.loads(Path('/long/logs/formal.json').read_text())['cases']['words-512'];case=old['request'];current=old['current_reference'];r={'settings':state(),'request':case,'current_reference':current,'current_source':'previous long-sequence/formal.json same frozen snapshot; formal phase will recompute current','selected':app.selected,'variants':{m:{'records':[]} for m in ('baseline','validate','attention','mlp','combined')}}
path=P/'logs/screen.json'
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
    app.use('baseline');t=time.perf_counter();baseline=app.invoke(case);r['baseline_reference']=baseline;r['first_baseline_seconds']=time.perf_counter()-t
    for mode in r['variants']:app.use(mode);app.invoke(case)
    r['handles_before']=len(app.engine.handles)
    for i in range(4):
        modes=list(r['variants']);modes=modes[i:]+modes[:i]
        for mode in modes:
            app.use(mode);before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(case);rec={'seconds':time.perf_counter()-t,'output':out,'vs_baseline':detailed(out,baseline),'vs_current':detailed(out,current),**app.counts(),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before};r['variants'][mode]['records'].append(rec);save()
            assert not rec['mlp_schedule']['errors'] and not any(x['errors'] for x in rec['long_kernels'].values()),(mode,rec)
            print('SCREEN',mode,round(rec['seconds'],4),rec['vs_baseline']['max_delta'],rec['vs_current']['max_delta'],rec['mlp_schedule']['parallel_layers'],flush=True)
    for v in r['variants'].values():v['timing']=stats([x['seconds'] for x in v['records']]);v['baseline_gate_passed']=all(x['vs_baseline']['passed'] for x in v['records']);v['current_gate_passed']=all(x['vs_current']['passed'] for x in v['records'])
    r['handles_after']=len(app.engine.handles);assert r['handles_before']==r['handles_after'];r['completed']=True;r['final_settings']=state();save()
finally:app.close()
