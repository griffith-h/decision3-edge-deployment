"""Selection between completed independent containers; no inference or hashing."""
from pathlib import Path
import json,statistics
p=Path(__file__).resolve().parent;unit=json.loads((p/'logs/grouped-unit.json').read_text());j=json.loads((p/'logs/screen-grouped.json').read_text());assert j['completed'] and unit['passed'];r={}
for mode in ('wide','grouped'):
 rows=[x for x in j['records'] if x['mode']==mode];assert len(rows)==8
 r[mode]={'median':statistics.median(x['seconds'] for x in rows),'passed':all(x['vs_current']['passed'] and x['vs_baseline']['passed'] and not x['full640_errors'] and not x['precise_errors'] and all(not v['errors'] and not v['native_fallback'] for v in x['long_kernels'].values()) for x in rows)}
assert r['wide']['passed'] or r['grouped']['passed'],'No qualified attention path'
selected=r['grouped']['passed'] and (not r['wide']['passed'] or r['grouped']['median']<r['wide']['median']*.98)
out={'grouped':selected,'selection':r,'rule':'All primary output/runtime gates and >=2% lower full-model median vs ungrouped wide path. All samples retained.'};(p/'execution.json').write_text(json.dumps(out,indent=2)+'\n');print('GROUP SELECTION',out)
