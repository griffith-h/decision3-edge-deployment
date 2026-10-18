from owner import *
from pathlib import Path
import argparse, resource
parser=argparse.ArgumentParser();parser.add_argument('--candidate',choices=['attention','scan','both'],default='both');selected=parser.parse_args().candidate
app=Owner();old=json.loads(Path('/row/logs/formal.json').read_text());screen=json.loads((P/'logs/screen.json').read_text())
requests=[(old['cases']['single']['request'],20,20)]
for key in ('words-410','words-512','words-921'):requests.append((screen['cases'][key]['request'],5,20))
cases=json.loads(Path('/mlp/http-cases.json').read_text())
natural=copy.deepcopy(next(c for c in cases if c['id']=='english'));natural.update(id='natural-long',state=(P/'natural-request.txt').read_text());natural['questions']={'difficulty':natural['questions']['difficulty']};requests.append((natural,3,10))
batch=copy.deepcopy(cases[0]);batch.update(id='batch-long',state=' '.join(['alpha']*500));batch['questions']={k:v for k,v in list(batch['questions'].items())[:2]};requests.append((batch,3,10))
boundary=copy.deepcopy(old['cases']['single']['request']);boundary.update(id='boundary-512',state=' '.join(['alpha']*409));requests.append((boundary,3,3))
requests.append((old['cases']['multi-wide']['request'],3,3))
requests.append((next(c for c in cases if c['id']=='image'),2,2))
path=P/'logs/formal.json';r={'settings':state(),'candidate':selected,'inventory':app.projections.inventory,'calibration':app.bank.calibration,'cases':{},'extra':[],'methodology':'Fresh current and row-fused references per input. Warmed interleaved baseline control blocks and candidate repeats; sample counts are per case. Current reference latency includes setup and is not a speed distribution.'}
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
def measure(c,mode,base,current):
    app.use(mode);before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(c);elapsed=time.perf_counter()-t
    return {'seconds':elapsed,'vs_baseline':detailed(out,base),'vs_current':detailed(out,current),**app.counts(),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before}
try:
    for c,ncontrol,ncandidate in requests:
        key=c['id'];row={'request':c,'variants':{m:{'records':[]} for m in ('baseline',selected)}};r['cases'][key]=row;print('FORMAL BEGIN',key,flush=True)
        app.use('current');t=time.perf_counter();current=app.invoke(c);row['current_reference']=current;row['current_first_seconds']=time.perf_counter()-t;save()
        app.use('baseline');t=time.perf_counter();base=app.invoke(c);row['baseline_reference']=base;row['baseline_first_seconds']=time.perf_counter()-t;row['baseline_vs_current']=detailed(base,current);row['input_tokens']=base['usage']['input_tokens'];save()
        for mode in ('baseline',selected):measure(c,mode,base,current)
        row['handles_before']=len(app.engine.handles);candidate_done=0
        for i in range(ncontrol):
            n=(ncandidate*(i+1)//ncontrol)-(ncandidate*i//ncontrol)
            modes=['baseline']+[selected]*n
            if i%2:modes.reverse()
            for mode in modes:
                rec=measure(c,mode,base,current);row['variants'][mode]['records'].append(rec);save()
                assert not any(v['errors'] or v['native_fallback'] for v in rec['long_kernels'].values()),(key,mode,rec)
                assert not rec['projections']['errors'],(key,mode,rec)
                if mode==selected and key in ('words-410','words-512','words-921','natural-long','batch-long'):
                    assert all(rec['long_kernels'][k]['optimized']==6 and not rec['long_kernels'][k]['guard_fallback'] for k in app.long.enabled),(key,mode,rec)
        for v in row['variants'].values():
            v['timing']=stats([x['seconds'] for x in v['records']]);v['baseline_gate_passed']=all(x['vs_baseline']['passed'] for x in v['records']);v['current_gate_passed']=all(x['vs_current']['passed'] for x in v['records'])
        row['handles_after']=len(app.engine.handles);assert row['handles_after']==row['handles_before'];row['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        row['latency_reduction']=1-row['variants'][selected]['timing']['median']/row['variants']['baseline']['timing']['median'];save()
        print('FORMAL',key,[(m,v['timing']['median'],v['baseline_gate_passed'],v['current_gate_passed']) for m,v in row['variants'].items()],row['latency_reduction'],flush=True)
        if key=='words-512':
            row['profile']=[];app.long.profile=True
            try:
                for _ in range(3):
                    for mode in ('baseline',selected):row['profile'].append({'mode':mode,**measure(c,mode,base,current)})
            finally:app.long.profile=False
            save()
    for c in cases[7:]:
        app.use('baseline');base=app.invoke(c)
        for _ in range(2):
            rec=measure(c,selected,base,base);r['extra'].append({'id':c['id'],**rec});assert rec['vs_baseline']['max_delta']==0 and rec['vs_baseline']['confidence_max_delta']==0
        save()
    r['completed']=True;r['native_paths_passed']=True;r['final_settings']=state();save()
finally:app.close()
