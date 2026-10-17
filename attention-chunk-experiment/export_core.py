import torch,onnx,json
from pathlib import Path
torch.set_num_threads(3)
class Core(torch.nn.Module):
    def forward(self,q,k,v,mask):
        k=k.repeat_interleave(4,dim=1);v=v.repeat_interleave(4,dim=1)
        att=torch.softmax((q@k.transpose(-1,-2))*.0625+mask,-1)
        return (att@v).transpose(1,2)
args=(torch.randn(1,8,115,256),torch.randn(1,2,115,256),torch.randn(1,2,115,256),torch.zeros(1,1,1,115))
torch.onnx.export(Core().eval(),args,'/work/attention-core.onnx',input_names=['q','k','v','mask'],output_names=['out'],opset_version=17)
onnx.checker.check_model(onnx.load('/work/attention-core.onnx'));print('EXPORTED core',flush=True)
