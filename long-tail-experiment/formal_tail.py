from runner import *
from pathlib import Path
import statistics,resource
app=Owner();previous=json.loads(Path('/long/logs/formal.json').read_text());c=previous['cases']['words-512']['request']
r={'settings':state(),'pilot':{},'cases':{},'extra':[],'selected_native':app.selected,'methodology':'615 target recomputes current and verifies equality with prior frozen reference. Other cases reuse explicitly identified prior current outputs; baseline and all candidates always infer afresh. Pilot6/mode,615 formal20 pairs,11510 pairs,others5/3/2 as recorded.'};path=P/'logs/formal.json'
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
def measure(case,mode,base,current):
    app.use(mode);before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(case);seconds=time.perf_counter()-t
    rec={'seconds':seconds,'output':out,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts(),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before}
    assert not rec['mlp_schedule']['errors'] and not rec['projections']['errors'] and not any(v['errors'] or v['native_fallback'] for v in rec['long_kernels'].values()),(case['id'],mode,rec)
    return rec
try:
    app.use('current');fresh_current=app.invoke(c);assert fresh_current==previous['cases']['words-512']['current_reference'];r['fresh_current_matches_prior']=True
    app.use('baseline');base=app.invoke(c);r['pilot_reference']={'baseline':base,'current':fresh_current}
    for m in ('baseline','validate_mlp','combined'):app.use(m);app.invoke(c);r['pilot'][m]={'records':[]}
    for i in range(6):
        modes=['baseline','validate_mlp','combined'];modes=modes[i%3:]+modes[:i%3]
        for m in modes:r['pilot'][m]['records'].append(measure(c,m,base,fresh_current));save()
    for v in r['pilot'].values():v['timing']=stats([x['seconds'] for x in v['records']]);v['passed']=all(x['vs_baseline']['passed'] and x['vs_current']['passed'] for x in v['records'])
    passing=[m for m in ('validate_mlp','combined') if r['pilot'][m]['passed']]
    assert passing,'No qualified candidate; keep failures and investigate'
    selected=min(passing,key=lambda m:r['pilot'][m]['timing']['median'])
    if selected=='combined' and 'validate_mlp' in passing and r['pilot']['combined']['timing']['median']>=.98*r['pilot']['validate_mlp']['timing']['median']:selected='validate_mlp'
    r['candidate']=selected;r['selection_rule']='Fastest passing pilot; prefer original attention when combined is less than2% faster';save();print('PILOT',[(m,v['timing']['median'],v['passed']) for m,v in r['pilot'].items()],'SELECTED',selected,flush=True)
    keys=[('words-512',20),('single',10),('words-410',5),('words-921',5),('natural-long',5),('batch-long',5),('boundary-512',3),('multi-wide',3),('image',2)]
    for key,n in keys:
        case=previous['cases'][key]['request'];current=fresh_current if key=='words-512' else previous['cases'][key]['current_reference']
        row={'request':case,'current_reference':current,'current_reference_source':'fresh and matches prior' if key=='words-512' else 'prior long-sequence/formal.json same frozen model/reference path','variants':{m:{'records':[]} for m in ('baseline',selected)}};r['cases'][key]=row;print('FORMAL BEGIN',key,flush=True)
        app.use('baseline');base=app.invoke(case);row['baseline_reference']=base;row['input_tokens']=base['usage']['input_tokens']
        for mode in ('baseline',selected):measure(case,mode,base,current)
        row['handles_before']=len(app.engine.handles)
        for i in range(n):
            for mode in (('baseline',selected) if i%2==0 else (selected,'baseline')):row['variants'][mode]['records'].append(measure(case,mode,base,current));save()
        for v in row['variants'].values():v['timing']=stats([x['seconds'] for x in v['records']]);v['baseline_gate_passed']=all(x['vs_baseline']['passed'] for x in v['records']);v['current_gate_passed']=all(x['vs_current']['passed'] for x in v['records'])
        row['handles_after']=len(app.engine.handles);assert row['handles_before']==row['handles_after'];row['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;row['reduction']=1-row['variants'][selected]['timing']['median']/row['variants']['baseline']['timing']['median'];save()
        print('FORMAL',key,[(m,v['timing']['median'],v['baseline_gate_passed'],v['current_gate_passed']) for m,v in row['variants'].items()],flush=True)
    for case in json.loads(Path('/mlp/http-cases.json').read_text())[7:]:
        app.use('baseline');base=app.invoke(case)
        for _ in range(2):
            rec=measure(case,selected,base,base);r['extra'].append({'id':case['id'],**rec});assert rec['vs_baseline']['all_numeric_max_delta']==0
        save()
    r['completed']=True;r['final_settings']=state();save()
finally:app.close()
