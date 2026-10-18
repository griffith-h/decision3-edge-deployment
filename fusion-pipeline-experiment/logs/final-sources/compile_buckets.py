"""Compile original, unquantized row-wise MLPs with public RKNN Toolkit2."""
import argparse,json,time,traceback
from pathlib import Path
import torch
from rknn.api import RKNN
p=argparse.ArgumentParser();p.add_argument('--rows',type=int,nargs='+',default=[128,384]);p.add_argument('--layers',type=int,nargs='+',default=list(range(12)));a=p.parse_args()
root=Path('/experiment');weights=torch.load('/weights/mlp-layers.pt',map_location='cpu',weights_only=False)
class MLP(torch.nn.Module):
 def __init__(self,row):
  super().__init__()
  for name,w in row['weights'].items():self.register_buffer(name,w.float())
 def forward(self,x):return torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x,self.gate_proj))*torch.nn.functional.linear(x,self.up_proj),self.down_proj)
for rows in a.rows:
 assert 0<rows<=2048
 for i in a.layers:
  name=f'mlp-{i}-rows{rows}';m=MLP(weights[i]).eval();x=torch.zeros(1,rows,1024)
  result={'name':name,'rows':rows,'layer':i,'quantized':False,'target':'rk3576','weights':'original BF16 promoted to FP32, no pruning'}
  t=time.perf_counter()
  with torch.no_grad():torch.onnx.export(m,(x,),root/(name+'.onnx'),input_names=['x'],output_names=['y'],opset_version=18,do_constant_folding=True)
  r=RKNN(verbose=True,verbose_file=str(root/'logs'/('rknn-'+name+'.log')))
  try:
   result['config_return']=r.config(target_platform='rk3576',optimization_level=3)
   result['load_return']=r.load_onnx(model=str(root/(name+'.onnx')))
   if result['load_return']==0:
    result['build_return']=r.build(do_quantization=False)
    if result['build_return']==0:result['export_return']=r.export_rknn(str(root/(name+'.rknn')))
  except Exception:result['error']=traceback.format_exc()
  finally:r.release()
  result['seconds']=time.perf_counter()-t;(root/'logs'/('compile-'+name+'.json')).write_text(json.dumps(result,indent=2)+'\n');print('COMPILED',result,flush=True)
  assert result.get('export_return')==0,result
