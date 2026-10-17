from baseline import stats
from mlp_graph import Graph,HalfGraph,lib
import torch,ctypes,time,json
from pathlib import Path
rows=torch.load('/work/mlp-layers.pt',weights_only=False);r={'layers':[]};torch.set_num_threads(3)
with torch.inference_mode():
 for i,row in enumerate(rows):
  p=Path('/work')/('mlp-'+str(i)+'-fp16graph.rknn');g=Graph(p);h=HalfGraph(p)
  try:
   x=row['input'].contiguous();a=torch.from_numpy(g.run(x.float().numpy())).bfloat16();b=h.run_bf16(x);torch.testing.assert_close(a,b,atol=.008,rtol=.008)
   meta=(ctypes.c_int*5)();assert lib.mlp_graph_meta(h.handle,meta)==0
   times=[]
   for _ in range(10):t=time.perf_counter();b=h.run_bf16(x);times.append(time.perf_counter()-t)
   r['layers'].append({'index':i,'meta':list(meta),'different':int((a!=b).sum()),'max_delta':(a.float()-b.float()).abs().max().item(),'timing':stats(times)})
   print('MLP HALF IO',i,r['layers'][-1]['max_delta'],r['layers'][-1]['timing']['median'],flush=True)
  finally:g.close();h.close()
 r['passed']=True
Path('/work/logs/mlp-half-io-unit.json').write_text(json.dumps(r,indent=2)+'\n')
