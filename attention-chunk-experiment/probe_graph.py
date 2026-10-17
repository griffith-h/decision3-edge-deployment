from graph import Graph
import numpy as np,json,time,statistics
from pathlib import Path
records=[]
for i in range(6):
    path=Path(f'/work/attention-{i}-fp16graph.rknn')
    if not path.exists():continue
    with np.load(f'/work/attention-{i}-replay.npz') as data:inputs={k:data[k] for k in ('x','cos','sin','mask')};ref=data['reference'].copy()
    g=Graph(path)
    try:
        out=g.run(inputs);assert np.isfinite(out).all();times=[]
        for _ in range(20):
            t=time.perf_counter();out=g.run(inputs);times.append(time.perf_counter()-t);assert np.isfinite(out).all()
        row={'layer':i,'times':times,'median':statistics.median(times),'max_abs':float(np.max(np.abs(out-ref))),'mean_abs':float(np.mean(np.abs(out-ref))),'cosine':float(np.dot(out.flatten().astype(np.float64),ref.flatten())/(np.linalg.norm(out)*np.linalg.norm(ref))),'inputs':g.names}
        # A future token must influence earlier queries (noncausal full attention).
        changed={k:v.copy() for k,v in inputs.items()};changed['x'][:,-1]*=-1;future=g.run(changed)
        row['future_changes_first_token']=float(np.max(np.abs(future[:,0]-out[:,0])));assert row['future_changes_first_token']>0
        records.append(row);print(row,flush=True)
    finally:g.close()
Path('/work/logs/graph-replay.json').write_text(json.dumps({'records':records,'passed':True},indent=2)+'\n')
