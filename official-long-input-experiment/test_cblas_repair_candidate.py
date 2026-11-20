import types,unittest
import torch
from sparse_rounding import SparseRepair
from cblas_repair_candidate import install

class ApproximationBoundaryTests(unittest.TestCase):
    def test_exact_simple_input_and_blas_threads_preserved(self):
        module=torch.nn.Linear(32,4,bias=False,dtype=torch.bfloat16)
        with torch.no_grad():module.weight.zero_();module.weight[:,0]=1
        x=torch.zeros(8,32,dtype=torch.bfloat16);x[:,0]=1
        state=SparseRepair(backend='cblas',noise_factor=.01,max_fraction=1)
        threads=state.blas_repair.threads();out=state.repair(module,x,torch.full((8,4),1.0039064))
        self.assertTrue(torch.equal(out,torch.ones(8,4,dtype=torch.bfloat16)))
        self.assertEqual(state.blas_repair.threads(),threads)
    def test_out_of_bounds_rejected_before_private_output_mutation(self):
        module=torch.nn.Linear(32,4,bias=False,dtype=torch.bfloat16);x=torch.ones(8,32,dtype=torch.bfloat16)
        state=SparseRepair(backend='cblas');out=torch.full((8,4),77.,dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'BLAS repair failed'):
            state.blas_repair.run(module,x,torch.tensor([[-1,0]],dtype=torch.long),out)
        self.assertTrue(bool((out==77).all()))
    def test_installer_restores_original_repair_and_clears_own_cache(self):
        previous=object();owner=types.SimpleNamespace(medium_sparse=types.SimpleNamespace(repair=previous))
        close=install(owner);candidate=owner.medium_sparse.repair;candidate.cache[1]='private'
        close();close();self.assertIs(owner.medium_sparse.repair,previous);self.assertEqual(candidate.cache,{})

if __name__=='__main__':unittest.main()
