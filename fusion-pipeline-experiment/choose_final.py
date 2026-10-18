from pathlib import Path
import json,statistics
p=Path(__file__).resolve().parent;j=json.loads((p/'logs/screen-scan-options.json').read_text());assert j['completed'];scores={}
for mode in sorted({x['mode'] for x in j['records']} - {'baseline'}):
 rows=[x for x in j['records'] if x['mode']==mode]
 passed=all(x['vs_current']['passed'] and x['vs_baseline']['passed'] and not x['full640_errors'] and not x['precise_errors'] and not x['mlp']['errors'] and not x['mlp']['range_fallback'] and not x['projections']['runtime_fallback'] and not x['projections']['range_fallback'] and all(not v['errors'] and not v['native_fallback'] for v in x['long_kernels'].values()) for x in rows)
 if mode=='neon':passed=passed and json.loads((p/'logs/neon-scan-unit.json').read_text())['passed']
 scores[mode]={'median':statistics.median(x['seconds'] for x in rows),'passed':passed,'n':len(rows)}
passing=[mode for mode,row in scores.items() if row['passed']];assert passing
mode=min(passing,key=lambda m:scores[m]['median'])
if mode!='wide' and scores['wide']['passed'] and scores[mode]['median']>=scores['wide']['median']*.98:mode='wide'
out=json.loads((p/'execution.json').read_text());out.update(mode=mode,residual=False,scan_selection=scores,omp_thread_limit=4 if mode in ('four','neon') else 3,scope='Frozen model B1/T615; other inputs retain qualified bypass and short multi-batch precision repair',rule='All primary current/incremental/runtime gates; require at least2% median gain over wide to add a scan variant. No samples discarded.')
(p/'execution.json').write_text(json.dumps(out,indent=2)+'\n');print('FINAL SELECTION',out)
