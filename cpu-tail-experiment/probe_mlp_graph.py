from baseline import stats
from mlp_graph import Graph
import torch,json,time
from pathlib import Path
r={'layers':[]};rows=torch.load('/work/mlp-layers.pt',weights_only=False);torch.set_num_threads(3)
for i,row in enumerate(rows):
 g=Graph(Path('/work')/('mlp-'+str(i)+'-fp16graph.rknn'))
 try:
  x=row['input'].float().contiguous().numpy();g.run(x);times=[]
  for _ in range(8):t=time.perf_counter();out=g.run(x);times.append(time.perf_counter()-t)
  actual=torch.from_numpy(out).float();ref=row['output'].float();rel=((actual-ref).norm()/ref.norm()).item()
  assert torch.isfinite(actual).all() and rel<.2,(i,rel)
  changed=x.copy();changed[:,-1]+=.1;other=g.run(changed)
  assert bool((out[:,:-1]==other[:,:-1]).all()),'unexpected token mixing'
  r['layers'].append({'index':i,'name':row['name'],'relative_l2':rel,'max_abs_delta':(actual-ref).abs().max().item(),'token_independence':True,'timing':stats(times)})
  print('MLP REPLAY',i,rel,r['layers'][-1]['timing']['median'],flush=True)
 finally:g.close()
r['passed']=True;Path('/work/logs/mlp-graph-replay.json').write_text(json.dumps(r,indent=2)+'\n')
