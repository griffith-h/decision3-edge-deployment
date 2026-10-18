"""Hardware sanity, row isolation, raw precision, range guards, warm primitive times."""
from experiment import *
import torch,ctypes
from bucket_graph import HalfGraph
from mlp_graph import HalfGraph as OldGraph
weights=torch.load(P/'mlp-layers.pt',map_location='cpu',weights_only=False)
r={'layers':{},'scope':'MLP primitive, no model probability claims'}
with torch.inference_mode():
 for i in [0,5,11]:
  old=OldGraph(f'/work/mlp-{i}-fp16graph.rknn');gs={n:HalfGraph(P/f'mlp-{i}-rows{n}.rknn') for n in (128,384)}
  try:
   x=weights[i]['input'].bfloat16();ref=old.run_bf16(x);row={};r['layers'][str(i)]=row
   for n,g in gs.items():
    z=torch.zeros(1,n,1024,dtype=torch.bfloat16);z[:,:115]=x;out=g.run_bf16(z);assert torch.count_nonzero(out[:,115:])==0
    delta=(out[:,:115].float()-ref.float()).abs();swapped=z.flip(1).contiguous();rev=g.run_bf16(swapped).flip(1);assert torch.equal(rev,out),'token cross-talk or non-determinism'
    for value in [float('nan'),float('inf'),70000.]:
     invalid=z.clone();invalid[0,0,0]=value
     try:g.run_bf16(invalid);raise AssertionError('accepted invalid input')
     except ValueError:pass
    # Buffer remains usable after rejection.
    assert torch.equal(g.run_bf16(z),out)
    for _ in range(4):g.run_bf16(z)
    times=[]
    for _ in range(30):
     t=time.perf_counter();g.run_bf16(z);times.append(time.perf_counter()-t)
    row[str(n)]={'timing':stats(times),'vs115_max_hidden_delta':float(delta.max()),'vs115_mean_hidden_delta':float(delta.mean()),'zero_padding_exact_zero':True,'row_permutation_exact':True,'range_rejection_and_recovery':True}
   print('PRIMITIVE',i,row,flush=True)
  finally:old.close();[g.close() for g in gs.values()]
r['passed']=True;(P/'logs/bucket-primitive.json').write_text(json.dumps(r,indent=2)+'\n')
