import types,unittest
import torch
from sparse_cpu_backend import Preparation
from rounding_band25_candidate import install

class Band25Tests(unittest.TestCase):
    def test_three_measured_omitted_boundaries_enter_native_mask(self):
        preparation=Preparation(vector_mask=True)
        for raw,norm,weight_norm in ((-4.890620231628418,42.81655502319336,.5436124205589294),
                                     (-2.320317029953003,42.02229309082031,.45652294158935547),
                                     (9.656241416931152,43.84358215332031,.9016417264938354)):
            value=torch.tensor([[raw]],dtype=torch.float32);xn=torch.tensor([norm]);wn=torch.tensor([weight_norm]);zero=torch.zeros(1)
            self.assertEqual(len(preparation.mask(value,xn,zero,xn,wn,zero,2e-7,1.,1)),0)
            self.assertTrue(torch.equal(preparation.mask(value,xn,zero,xn,wn,zero,2.5e-7,1.,1),torch.tensor([[0,0]])))
    def test_scope_preserves_limits_and_restores_preceding_band(self):
        state=types.SimpleNamespace(noise_factor=2e-7,max_fraction=.1,max_entries=65536,counts={'noise_factor':2e-7})
        close=install(types.SimpleNamespace(medium_sparse=types.SimpleNamespace(repair=state)))
        self.assertEqual(state.noise_factor,2.5e-7);self.assertEqual((state.max_fraction,state.max_entries),(.1,65536))
        close();close();self.assertEqual(state.noise_factor,2e-7);self.assertEqual(state.counts,{'noise_factor':2e-7})

if __name__=='__main__':unittest.main()
