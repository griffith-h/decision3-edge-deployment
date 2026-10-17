from baseline import stats
from mlp_graph import HalfGraph,lib
import torch,json,time
from pathlib import Path
rows=torch.load('/work/mlp-layers.pt',weights_only=False);r={'layers':[]};torch.set_num_threads(3)
with torch.inference_mode():
 for i in (0,5,11):
  g=HalfGraph(Path('/work')/('mlp-'+str(i)+'-fp16graph.rknn'));x=rows[i]['input'];row={'index':i,'modes':{}};r['layers'].append(row)
  try:
   ref=g.run_bf16(x)
   for mode in (0,1,2,3,0):
    rc=lib.mlp_graph_set_core(g.handle,mode);label=str(mode) if str(mode) not in row['modes'] else '0-after';item={'return':rc};row['modes'][label]=item
    if rc:lib.mlp_graph_set_core(g.handle,0);continue
    g.run_bf16(x);times=[]
    for _ in range(10):t=time.perf_counter();out=g.run_bf16(x);times.append(time.perf_counter()-t)
    item['timing']=stats(times);item['max_delta']=(out.float()-ref.float()).abs().max().item();torch.testing.assert_close(out,ref,atol=.008,rtol=.008)
    print('GRAPH CORE',i,mode,rc,item['timing']['median'],flush=True)
  finally:g.close()
 supported=all(x['modes']['3']['return']==0 for x in r['layers'])
 r['combined_supported']=supported;r['worth_model_test']=supported and sum(x['modes']['3']['timing']['median'] for x in r['layers'])<.95*sum(x['modes']['0']['timing']['median'] for x in r['layers']);r['passed']=True
Path('/work/logs/mlp-core-probe.json').write_text(json.dumps(r,indent=2)+'\n');print('MULTICORE WORTH TEST',r['worth_model_test'],flush=True)
