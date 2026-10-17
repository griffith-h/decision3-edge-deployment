"""Export the entire original text attention module, including QKV and output projections."""
import argparse,json,time,torch,numpy as np
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--strict',action='store_true');p.add_argument('--layer',type=int,default=0);a=p.parse_args()
root=Path('/work');items=torch.load(root/'attention-layers.pt',weights_only=False);item=items[a.layer];torch.set_num_threads(3)
class Attention(torch.nn.Module):
    def __init__(self,item,strict):
        super().__init__();self.strict=strict;self.eps=item['eps'];self.scale=item['scaling']
        for key,v in item['weights'].items():self.register_buffer(key.replace('.','_'),v.float())
    def r(self,x):return x.to(torch.bfloat16).float() if self.strict else x
    def linear(self,x,w):
        x=torch.nn.functional.linear(x,w)
        return self.r(x.half().float()) if self.strict else x
    def norm(self,x,w):return self.r(x*torch.rsqrt((x*x).mean(-1,keepdim=True)+self.eps)*(1+w))
    def rope(self,x,cos,sin):
        r=x[...,:64];rot=torch.cat((-r[...,32:],r[...,:32]),-1)
        return torch.cat((self.r(self.r(r*cos)+self.r(rot*sin)),x[...,64:]),-1)
    def forward(self,x,cos,sin,mask):
        qg=self.linear(x,self.q_proj_weight).reshape(1,115,8,512);q,gate=qg.chunk(2,-1);gate=gate.reshape(1,115,2048)
        q=self.norm(q,self.q_norm_weight).transpose(1,2)
        k=self.norm(self.linear(x,self.k_proj_weight).reshape(1,115,2,256),self.k_norm_weight).transpose(1,2)
        v=self.linear(x,self.v_proj_weight).reshape(1,115,2,256).transpose(1,2)
        q=self.rope(q,cos[:,None],sin[:,None]);k=self.rope(k,cos[:,None],sin[:,None])
        k=k.repeat_interleave(4,dim=1);v=v.repeat_interleave(4,dim=1)
        att=torch.softmax(torch.matmul(q,k.transpose(-1,-2))*self.scale+mask,dim=-1)
        out=self.r(torch.matmul(att,v).transpose(1,2).reshape(1,115,2048))
        out=self.r(out*self.r(torch.sigmoid(gate)))
        return self.linear(out,self.o_proj_weight)
model=Attention(item,a.strict).eval();args=tuple(item[n].float() for n in ('x','cos','sin'))+((~item['mask']).float()*-10000,)
with torch.inference_mode():
    out=model(*args);ref=item['out'].float();diff=(out-ref).abs();cos=torch.nn.functional.cosine_similarity(out.flatten(),ref.flatten(),dim=0).item()
name=f"attention-{a.layer}-"+('strict' if a.strict else 'fp16graph');path=root/(name+'.onnx')
torch.onnx.export(model,args,str(path),input_names=['x','cos','sin','mask'],output_names=['out'],opset_version=17,do_constant_folding=True)
import onnx
m=onnx.load(path);onnx.checker.check_model(m)
report={'layer':item['name'],'strict_bf16_boundaries':a.strict,'exported':str(path),'ops':sorted(set(n.op_type for n in m.graph.node)),
        'cpu_reference_max_abs':diff.max().item(),'cpu_reference_mean_abs':diff.mean().item(),'cpu_reference_cosine':cos,'note':'activation diagnostic, not probability tolerance; non-strict changes arithmetic rounding, original weights and math/mask retained'}
(root/'logs'/('export-'+name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)
# Portable replay inputs and baseline outputs, no tensor pickle required by runtime.
np.savez(root/(f'attention-{a.layer}-replay.npz'),x=args[0].numpy(),cos=args[1].numpy(),sin=args[2].numpy(),mask=args[3].numpy(),reference=ref.numpy())
