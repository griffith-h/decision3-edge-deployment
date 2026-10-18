from pathlib import Path
import json,statistics
p=Path(__file__).resolve().parent;j=json.loads((p/'logs/ablate-mlp.json').read_text());assert j['completed'];scores={}
for enabled in (False,True):
 rows=[x for x in j['records'] if x['full640']==enabled];assert len(rows)==12;scores[str(enabled)]={'median':statistics.median(x['seconds'] for x in rows),'passed':all(x['vs_current']['passed'] and x['vs_baseline']['passed'] and not x['full640_errors'] and all(not y['errors'] and not y['native_fallback'] for y in x['long_kernels'].values()) for x in rows)}
assert scores['False']['passed'] or scores['True']['passed'],'No qualified MLP path'
selected=scores['True']['passed'] and (not scores['False']['passed'] or scores['True']['median']<scores['False']['median']*.98)
cfg=json.loads((p/'execution.json').read_text());cfg.update(full640=selected,mlp_ablation=scores,mlp_rule='Require at least2% full-model median improvement in final combined path to retain new640 graphs; otherwise original dual-handle128+128+384.');(p/'execution.json').write_text(json.dumps(cfg,indent=2)+'\n');print('MLP SELECTION',cfg)
