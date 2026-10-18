import sys
sys.path.insert(0,'/tail')
from runner import *
from bucket_graph import HalfGraph,lib
import ctypes
p=Path('/experiment');r={'settings':state(),'records':[],'scope':'Real-layer MLP replay only; no full model probability claim'};captured=torch.load('/tail/captured.pt',weights_only=False)['mlp']
with torch.inference_mode():
 for i in (0,5,11):
  graph=HalfGraph(p/f'mlp-{i}-rows640.rknn')
  try:
   x=captured[i]['input'];ref=captured[i]['reference'];padded=torch.zeros((1,640,1024),dtype=x.dtype);padded[:,:615]=x
   for mask in (0,1,2,3):
    assert lib.row_graph_set_core(graph.handle,mask)==0
    for _ in range(3):graph.run_bf16(padded)
    for rep in range(15):
     t=time.perf_counter();out=graph.run_bf16(padded);dt=time.perf_counter()-t;delta=(out[:,:615].float()-ref.float()).abs();r['records'].append({'layer':i,'mask':mask,'rep':rep,'seconds':dt,'max_hidden_delta':float(delta.max()),'mean_hidden_delta':float(delta.mean()),'padding_zero':int(torch.count_nonzero(out[:,615:]))==0})
    print('MLP640',i,mask,stats([x['seconds'] for x in r['records'] if x['layer']==i and x['mask']==mask]),r['records'][-1],flush=True)
  finally:lib.row_graph_set_core(graph.handle,0);graph.close()
r['completed']=True;(p/'logs/probe-mlp640.json').write_text(json.dumps(r,indent=2)+'\n')
