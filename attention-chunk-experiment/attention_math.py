import torch
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
