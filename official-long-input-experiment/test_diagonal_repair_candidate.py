import unittest,types
import torch
from sparse_cpu_backend import BlasRepair,DiagonalRepair
from diagonal_repair_candidate import install

class DiagonalRepairTests(unittest.TestCase):
    def test_pinned_blocks_match_previous_repair_on_all_entries(self):
        torch.manual_seed(44)
        for k,n in ((64,32),(1024,128),(2048,128),(6144,64),(10240,32)):
            layer=torch.nn.Linear(k,n,bias=False,dtype=torch.bfloat16);x=torch.randn(18,k,dtype=torch.bfloat16)[::2]
            positions=torch.ones(9,n,dtype=torch.bool).nonzero();expected=torch.empty(9,n,dtype=torch.bfloat16);out=torch.empty_like(expected)
            BlasRepair().run(layer,x,positions,expected);DiagonalRepair().run(layer,x,positions,out)
            self.assertTrue(torch.equal(out,expected),(k,n,int((out!=expected).sum())))
    def test_order_bounds_and_dtype_reject_before_output_mutation(self):
        layer=torch.nn.Linear(64,32,bias=False,dtype=torch.bfloat16);x=torch.ones(3,64,dtype=torch.bfloat16);out=torch.full((3,32),77.,dtype=torch.bfloat16);state=DiagonalRepair()
        for positions in (torch.tensor([[0,32]]),torch.tensor([[2,0],[0,0]])):
            with self.assertRaisesRegex(ValueError,'repair failed'):state.run(layer,x,positions,out)
            self.assertTrue(bool((out==77).all()))
        with self.assertRaisesRegex(ValueError,'tensors'):state.run(layer,x.float(),torch.tensor([[0,0]]),out)
    def test_partial_groups_unselected_and_installer_restoration(self):
        layer=torch.nn.Linear(64,32,bias=False,dtype=torch.bfloat16);x=torch.randn(3,64,dtype=torch.bfloat16);pairs=torch.tensor([[0,3],[0,7],[1,8],[2,9],[2,11]])
        expected=torch.full((3,32),77.,dtype=torch.bfloat16);out=expected.clone()
        BlasRepair().run(layer,x,pairs,expected);DiagonalRepair().run(layer,x,pairs,out);self.assertTrue(torch.equal(out,expected))
        old=object();state=types.SimpleNamespace(blas_repair=old,backend='cblas',counts={'backend':'cblas'});owner=types.SimpleNamespace(medium_sparse=types.SimpleNamespace(repair=state))
        close=install(owner);self.assertEqual(state.backend,'diagonal-q352');close();close();self.assertIs(state.blas_repair,old);self.assertEqual(state.backend,'cblas')
if __name__=='__main__':unittest.main()
