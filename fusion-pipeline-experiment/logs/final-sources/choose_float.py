"""Gate-first final choice, based on independent correctness and paired timing."""
from pathlib import Path
import json,statistics
p=Path(__file__).resolve().parent
screen=json.loads((p/'logs/screen-float.json').read_text());ablate=json.loads((p/'logs/ablate-float-final.json').read_text());assert screen['completed'] and ablate['completed'] and len(screen['cases'])==13
rank={}
for mode in ('float','float_readout'):
 rows=[r for r in ablate['records'] if r['mode']==mode];assert len(rows)==12
 correctness=[r for c in screen['cases'].values() for r in c['records'] if r['mode']==mode];assert len(correctness)==13
 passed=all(r['vs_current']['passed'] and not r['precise_errors'] and not r.get('last_readout_errors',[]) and not r['precise_range_fallbacks'] for r in rows+correctness)
 rank[mode]={'median':statistics.median(r['seconds'] for r in rows),'passed':passed,'n':len(rows)}
assert rank['float']['passed'],'FP32 full path remains unqualified'
readout=rank['float_readout']['passed'] and rank['float_readout']['median']<=rank['float']['median']*.98
e=json.loads((p/'execution.json').read_text());e.update(fp32=True,precise615=True,precise_width=640,readout=readout,full640=False,float_selection=rank,float_rule='All13 original-current gates plus12 paired samples; require2% median improvement to add private readout. Final34-case regression is still mandatory.',scope='Frozen model B1/T615 only for FP32/precise615/readout; existing qualified bypass for other shapes; short multi-batch precision repair retained.')
(p/'execution.json').write_text(json.dumps(e,indent=2)+'\n');print('FLOAT SELECTION',e)
