import unittest
from unittest.mock import patch
import torch
import test_native_preparation_candidate as original
from sparse_cpu_backend import Preparation

class VectorPreparationTests(original.NativePreparationTests):
    def setUp(self):
        self.factory=patch.object(original,'Preparation',lambda:Preparation(vector_mask=True));self.factory.start()
    def tearDown(self):self.factory.stop()
    def test_midpoint_lanes_signs_small_values_and_tail_match_scalar_mask(self):
        raw=(torch.arange(31*129,dtype=torch.int32)%131-65+0x3f808000).view(torch.float32).reshape(31,129)
        raw[::2]*=-1;raw[0,:4]=torch.tensor([0.,1e-38,-1e-38,1e-35])
        xn=torch.linspace(.01,3.,31);xl=xn*1e-9;hn=xn.clone();wn=torch.linspace(.2,2.,129);wl=wn*1e-9
        expected=Preparation().mask(raw,xn,xl,hn,wn,wl,1e-7,1.,65536)
        actual=Preparation(vector_mask=True).mask(raw,xn,xl,hn,wn,wl,1e-7,1.,65536)
        self.assertTrue(torch.equal(actual,expected))

if __name__=='__main__':unittest.main()
