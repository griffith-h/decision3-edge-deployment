import types,unittest
import torch
from sparse_cpu_backend import Preparation
from rounding_band_candidate import install

class BandTests(unittest.TestCase):
    def test_measured_omitted_boundary_enters_expanded_native_mask(self):
        p=Preparation(vector_mask=True);raw=torch.tensor([[2.9140608310699463]],dtype=torch.float32)
        xn=torch.tensor([39.8892936706543]);wn=torch.tensor([0.3644103407859802]);zero=torch.zeros(1)
        before=p.mask(raw,xn,zero,xn,wn,zero,1e-7,1.,1)
        after=p.mask(raw,xn,zero,xn,wn,zero,2e-7,1.,1)
        self.assertEqual(len(before),0);self.assertTrue(torch.equal(after,torch.tensor([[0,0]])))
    def test_installer_preserves_budgets_and_restores_previous_factor(self):
        state=types.SimpleNamespace(noise_factor=1e-7,max_fraction=.1,max_entries=65536,counts={})
        close=install(types.SimpleNamespace(medium_sparse=types.SimpleNamespace(repair=state)))
        self.assertEqual(state.noise_factor,2e-7);self.assertEqual((state.max_fraction,state.max_entries),(.1,65536))
        close();close();self.assertEqual(state.noise_factor,1e-7);self.assertEqual(state.counts,{})
if __name__=='__main__':unittest.main()
