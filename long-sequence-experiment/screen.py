from owner import *
import argparse
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true');args=parser.parse_args()
app=Owner();old=json.loads(Path('/row/logs/formal.json').read_text())
cases=[old['cases'][k]['request'] for k in ('words-410','words-512')]
if args.quick:cases=cases[:1]
else:
    c=copy.deepcopy(cases[0]);c.update(id='words-921',state=' '.join(['alpha']*921));cases.append(c)
path=P/'logs/screen.json';r={'settings':state(),'scope':'3 warmed samples/mode; fresh previous-row baseline; canonical current references from prior frozen run for 513/615 only','cases':{}}
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
try:
    for c in cases:
        key=c['id'];row={'request':c,'variants':{m:{'records':[]} for m in ('baseline','attention','scan','both')}};r['cases'][key]=row
        print('SCREEN BEGIN',key,flush=True)
        app.use('baseline');t=time.perf_counter();ref=app.invoke(c);row['baseline_reference']=ref;row['first_baseline_seconds']=time.perf_counter()-t;row['input_tokens']=ref['usage']['input_tokens']
        canonical=old['cases'].get(key,{}).get('reference');row['canonical_reference']=canonical
        for mode in row['variants']:app.use(mode);app.invoke(c)
        row['handles_before']=len(app.engine.handles)
        for i in range(3):
            modes=list(row['variants']);modes=modes[i:]+modes[:i]
            for mode in modes:
                app.use(mode);before=app.engine.cpu_fallbacks;t=time.perf_counter();out=app.invoke(c);rec={'seconds':time.perf_counter()-t,'vs_baseline':detailed(out,ref),'vs_current':detailed(out,canonical) if canonical else None,**app.counts(),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before};row['variants'][mode]['records'].append(rec);save()
                assert not any(v['errors'] for v in rec['long_kernels'].values()),rec
                print('SCREEN',key,mode,round(rec['seconds'],4),rec['vs_baseline']['max_delta'],rec['vs_current']['max_delta'] if canonical else None,{k:v['optimized'] for k,v in rec['long_kernels'].items()},flush=True)
        for v in row['variants'].values():v['timing']=stats([x['seconds'] for x in v['records']])
        row['handles_after']=len(app.engine.handles);assert row['handles_after']==row['handles_before'];save()
    r['completed']=True;r['final_settings']=state();save()
finally:app.close()
