from graph import Graph
from pathlib import Path
import numpy as np,json
rows=[]
for i in range(6):
    g=Graph(f'/work/attention-{i}-fp16graph.rknn')
    try:
        for label in ('all','left','last_only'):
            with np.load(f'/work/mask-{i}-{label}.npz') as d:inputs={k:d[k] for k in ('x','cos','sin','mask')};ref=d['reference']
            saved={k:v.copy() for k,v in inputs.items()};out=g.run(inputs)
            assert np.isfinite(out).all() and all(np.array_equal(v,saved[k]) for k,v in inputs.items())
            ratio=float(np.linalg.norm(out-ref)/np.linalg.norm(ref));assert ratio<.03,(i,label,ratio)
            rows.append({'layer':i,'mask':label,'relative_activation_l2_error':ratio,'max_abs':float(np.max(np.abs(out-ref)))})
    finally:g.close()
Path('/work/logs/graph-mask-unit.json').write_text(json.dumps({'passed':True,'checks':rows,'scope':'activation/mask operator check; distinct from .01 classifier probability acceptance'},indent=2)+'\n');print('PASS graph masks 18',flush=True)
