import unittest
import torch
from sparse_cpu_backend import Preparation

class NativePreparationTests(unittest.TestCase):
    def test_float64_norms_and_fp16_loss_include_small_values_and_strided_rows(self):
        torch.manual_seed(7);x=torch.randn(14,64,dtype=torch.bfloat16)[::2]
        x[0].zero_();x[0,0]=1e-38;x[1,0]=2**-20;x[1,1]=-2**-25
        norm,loss,hn=Preparation().norms(x,True);value=x.float();half=x.half().float()
        reference=lambda a:torch.linalg.vector_norm(a,dim=-1,dtype=torch.float64).float()
        self.assertTrue(torch.equal(norm,reference(value)));self.assertTrue(torch.equal(loss,reference(value-half)));self.assertTrue(torch.equal(hn,reference(half)))
    def test_mask_order_and_fp32_operations_match_original_torch(self):
        torch.manual_seed(9);raw=torch.randn(5,17);raw[0,0]=0.;raw[1,1]=1.00390625;raw[2,2]=-1.00390625
        xn=torch.rand(5)+1;xl=torch.rand(5)*1e-7;xhn=xn.clone();wn=torch.rand(17)+1;wl=torch.rand(17)*1e-7
        xn[0]=1e-35;xl[0]=0;xhn[0]=xn[0]
        band=(1e-7*xn[:,None]*wn[None,:]+xl[:,None]*wn[None,:])+xhn[:,None]*wl[None,:]
        bits=raw.abs().view(torch.int32);lo=(bits & -65536).view(torch.float32);hi=((bits & -65536)+65536).view(torch.float32)
        expected=((raw.abs()-(lo+hi)*.5).abs()<=band).nonzero()
        actual=Preparation().mask(raw,xn,xl,xhn,wn,wl,1e-7,1.,65536)
        self.assertTrue(torch.equal(actual,expected))
    def test_nonfinite_density_and_coefficient_guards(self):
        state=Preparation();raw=torch.zeros(3,4);x=torch.ones(3);w=torch.ones(4)
        with self.assertRaisesRegex(ValueError,'density budget'):state.mask(raw,x,x,x,w,w,1e-7,.1,1)
        raw[0,0]=float('nan')
        with self.assertRaisesRegex(ValueError,'Nonfinite'):state.mask(raw,x,x,x,w,w,1e-7,1.,65536)
        with self.assertRaisesRegex(ValueError,'coefficient'):state.mask(torch.zeros(3,4),x[:2],x,x,w,w,1e-7,1.,65536)
        with self.assertRaisesRegex(ValueError,'finite/range'):state.norms(torch.full((2,32),65536.,dtype=torch.bfloat16))

if __name__=='__main__':unittest.main()
