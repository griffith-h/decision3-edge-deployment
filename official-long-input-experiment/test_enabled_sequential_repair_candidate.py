"""Original CPU numeric check and backend-change rejection before mutation."""
import unittest
from types import SimpleNamespace
import torch
import torch.nn.functional as f
from enabled_sequential_repair_candidate import EnabledSequentialRepair,install

class EnabledSequentialTests(unittest.TestCase):
    def setUp(self):
        self.old=(torch.backends.mkldnn.enabled,torch.get_num_threads(),torch.get_float32_matmul_precision())
        torch.backends.mkldnn.enabled=True;torch.set_num_threads(3);torch.set_float32_matmul_precision('highest')
    def tearDown(self):
        torch.backends.mkldnn.enabled=self.old[0];torch.set_num_threads(self.old[1]);torch.set_float32_matmul_precision(self.old[2])
    def test_actual_enabled_fp32_reference_and_untouched_positions(self):
        generator=torch.Generator().manual_seed(68)
        for k,n in ((1024,128),(2048,128),(3584,128)):
            x=torch.randn(3,43,k,generator=generator,dtype=torch.bfloat16)
            module=SimpleNamespace(weight=torch.randn(n,k,generator=generator,dtype=torch.bfloat16))
            expected=f.linear(x.float(),module.weight.float()).bfloat16().reshape(-1,n)
            mask=torch.rand(129,n,generator=generator)<.15;out=torch.full_like(expected,77.)
            EnabledSequentialRepair().run(module,x.reshape(-1,k),mask.nonzero(),out)
            self.assertTrue(torch.equal(out[mask],expected[mask]),k)
            self.assertTrue(bool((out[~mask]==77).all()))
    def test_changed_backend_rejects_before_output_mutation_and_scope_restores(self):
        previous=object();state=SimpleNamespace(blas_repair=previous,backend='previous',counts={'backend':'previous'})
        owner=SimpleNamespace(medium_sparse=SimpleNamespace(repair=state));close=install(owner)
        candidate=state.blas_repair;output=torch.full((3,32),77.,dtype=torch.bfloat16)
        module=SimpleNamespace(weight=torch.ones(32,64,dtype=torch.bfloat16));x=torch.ones(3,64,dtype=torch.bfloat16)
        torch.backends.mkldnn.enabled=False
        with self.assertRaisesRegex(ValueError,'backend changed'):
            candidate.run(module,x,torch.tensor([[0,0]]),output)
        self.assertTrue(bool((output==77).all()))
        close();close();self.assertIs(state.blas_repair,previous);self.assertEqual(state.counts['backend'],'previous')

if __name__=='__main__':unittest.main()
