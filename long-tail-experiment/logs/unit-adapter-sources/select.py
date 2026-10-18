from pathlib import Path
import json
P=Path('/experiment');r=json.loads((P/'logs/replay.json').read_text());assert r['completed']
selected={};details={}
for kind in ('attention','scan'):
    baseline=r[kind]['baseline']['sum_layer_medians'];ranking=r[kind+'_ranking'];winner=next(name for name in ranking if r[kind][name]['passed'])
    candidate=r[kind][winner];gain=1-candidate['sum_layer_medians']/baseline
    if winner=='baseline' or gain<.02:
        selected[kind]={'kind':'original'} if kind=='scan' else None
        details[kind]={'selected':'baseline','best_name':winner,'gain':gain,'reason':'No passing candidate >=2% faster in component replay'}
    else:selected[kind]=candidate['config'];details[kind]={'selected':winner,'gain':gain,'reason':'Replay-only filter; still needs end-to-end validation'}
assert selected['attention'] is not None,'No attention candidate; do not force a regression'
(P/'logs/raw-selected.json').write_text((P/'selected.json').read_text())
(P/'selected.json').write_text(json.dumps(selected,indent=2)+'\n');(P/'logs/selection.json').write_text(json.dumps(details,indent=2)+'\n');print('SELECTION',selected,details,flush=True)
