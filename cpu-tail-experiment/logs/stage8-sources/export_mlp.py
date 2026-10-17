import argparse,torch,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--layer',type=int,required=True);a=p.parse_args();root=Path('/work');rows=torch.load(root/'mlp-layers.pt',map_location='cpu',weights_only=False);row=rows[a.layer]
class MLP(torch.nn.Module):
 def __init__(self):
  super().__init__()
  for name,w in row['weights'].items():self.register_buffer(name,w.float())
 def forward(self,x):return torch.nn.functional.linear(torch.nn.functional.silu(torch.nn.functional.linear(x,self.gate_proj))*torch.nn.functional.linear(x,self.up_proj),self.down_proj)
m=MLP().eval();x=row['input'].float();name='mlp-'+str(a.layer)+'-fp16graph'
with torch.no_grad():torch.onnx.export(m,(x,),root/(name+'.onnx'),input_names=['x'],output_names=['y'],opset_version=18,do_constant_folding=True)
print('EXPORTED',name,row['name'],flush=True)
