# -*- coding: utf-8 -*-
"""Independent audit of the confidence field skipped by historical compare()."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
selected=json.loads((R/'cpu-tail-experiment/logs/formal-graph-single.json').read_text())
refs=json.loads((R/'attention-chunk-experiment/logs/final-single-scan.json').read_text())['cases']
extra=json.loads((R/'cpu-tail-experiment/logs/lut-screen.json').read_text())['extra_refs']
rows={}
for group in ('cases','extra'):
 for k,v in selected[group].items():
  ref=refs[k]['response'] if group=='cases' else extra[k]
  delta={name:abs(answer['confidence']-ref['answers'][name]['confidence']) for name,answer in v['response']['answers'].items() if 'confidence' in answer}
  rows[k]={'confidence_deltas':delta,'max_delta':max(delta.values(),default=0)}
r={'cases':rows,'max_confidence_delta_vs_442':max(x['max_delta'] for x in rows.values()),'historical_compare_excluded_confidence':True,'not_part_of_historical_001_gate':True,'task_accuracy_not_evaluated':True}
(P/'confidence-audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print('CONFIDENCE AUDIT',r['max_confidence_delta_vs_442'])
