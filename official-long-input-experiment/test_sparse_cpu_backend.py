import types,unittest
import torch
from sparse_cpu_backend import RowGather

class RowGatherTests(unittest.TestCase):
    def test_noncontiguous_positions_partial_tile_and_strided_source_rows(self):
        kernel=RowGather();x=(torch.arange(16*32).reshape(16,32)%13).bfloat16()[::2]
        module=types.SimpleNamespace(weight=(torch.arange(16*32).reshape(16,32)%7).bfloat16()[::2])
        positions=torch.tensor([[0,3,7],[1,5,6]],dtype=torch.long).t()
        a,b=kernel.run(module,x,positions)
        self.assertTrue(torch.equal(a[:3],x[positions[:,0]].float()));self.assertTrue(torch.equal(b[:3],module.weight[positions[:,1]].float()))
        self.assertTrue(torch.equal(a[3],a[2]));self.assertTrue(torch.equal(b[3],b[2]))
    def test_wrong_dtype_and_out_of_bounds_rejected_before_result(self):
        kernel=RowGather();x=torch.ones(4,32,dtype=torch.bfloat16);module=types.SimpleNamespace(weight=torch.ones(8,32,dtype=torch.bfloat16))
        with self.assertRaisesRegex(ValueError,'Unsupported'):kernel.run(module,x,torch.tensor([[0,0]],dtype=torch.int32))
        with self.assertRaisesRegex(ValueError,'gather failed'):kernel.run(module,x,torch.tensor([[-1,0]],dtype=torch.long))
        self.assertTrue(bool((x==1).all()));self.assertTrue(bool((module.weight==1).all()))
    def test_empty_gather_returns_empty_tensors_without_dereference(self):
        kernel=RowGather();x=torch.ones(4,32,dtype=torch.bfloat16);module=types.SimpleNamespace(weight=torch.ones(8,32,dtype=torch.bfloat16))
        a,b=kernel.run(module,x,torch.empty((0,2),dtype=torch.long))
        self.assertEqual(a.shape,(0,32));self.assertEqual(b.shape,(0,32))

if __name__=='__main__':unittest.main()
