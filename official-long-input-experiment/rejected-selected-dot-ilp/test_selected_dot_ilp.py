import types,unittest
import torch
from enabled_sequential_repair_candidate import EnabledSequentialRepair
from selected_dot_ilp import SelectedDot

class SelectedDotTests(unittest.TestCase):
    def setUp(self):
        self.context=torch.backends.mkldnn.flags(enabled=True);self.context.__enter__()
        self.previous_threads=torch.get_num_threads();self.previous_precision=torch.get_float32_matmul_precision()
        torch.set_num_threads(3);torch.set_float32_matmul_precision('highest')
    def tearDown(self):
        torch.set_num_threads(self.previous_threads);torch.set_float32_matmul_precision(self.previous_precision)
        self.context.__exit__(None,None,None)
    def test_original_bf16_and_continuous_fp32_order_all_widths_strides_tails(self):
        torch.set_num_threads(3);g=torch.Generator().manual_seed(88);old=EnabledSequentialRepair()
        with torch.inference_mode():
            for k,n in ((4,19),(64,31),(1024,128),(2048,128),(3584,128),(6144,32),(10240,32)):
                x=torch.randn(7,k+8,dtype=torch.bfloat16,generator=g)[:,:k]
                weight=torch.randn(n,k+8,dtype=torch.bfloat16,generator=g)[:,:k];m=types.SimpleNamespace(weight=weight)
                pairs=torch.tensor([[r,c] for r,count in ((0,1),(2,5),(3,9),(6,17)) for c in range(count)],dtype=torch.long)
                expected=torch.full((7,n),77.,dtype=torch.bfloat16);old.run(m,x,pairs,expected)
                raw4=None
                for width in (4,8,16):
                    out=torch.full_like(expected,77.);raw=torch.empty(len(pairs),dtype=torch.float32)
                    SelectedDot(width).run(m,x,pairs,out,raw)
                    self.assertTrue(torch.equal(out,expected),(k,width))
                    if raw4 is None:raw4=raw.clone()
                    else:self.assertTrue(torch.equal(raw.view(torch.int32),raw4.view(torch.int32)),(k,width))
    def test_empty_repeated_points_and_invalid_indices_rejected(self):
        torch.set_num_threads(3);m=types.SimpleNamespace(weight=torch.ones(32,8,dtype=torch.bfloat16));x=torch.ones(3,8,dtype=torch.bfloat16)
        for width in (8,16):
            kernel=SelectedDot(width);out=torch.full((3,32),77.,dtype=torch.bfloat16)
            kernel.run(m,x,torch.empty((0,2),dtype=torch.long),out);self.assertTrue(bool((out==77).all()))
            kernel.run(m,x,torch.tensor([[0,2],[0,2],[2,31]]),out);self.assertEqual(float(out[0,2]),8.)
            for pairs in (torch.tensor([[1,0],[0,0]]),torch.tensor([[3,0]]),torch.tensor([[0,32]])):
                with self.assertRaises(ValueError):kernel.run(m,x,pairs,out)
    def test_nonfinite_output_and_wrong_backend_are_rejected(self):
        torch.set_num_threads(3);m=types.SimpleNamespace(weight=torch.full((16,8),float('inf'),dtype=torch.bfloat16));x=torch.ones(1,8,dtype=torch.bfloat16)
        kernel=SelectedDot(8);out=torch.zeros(1,16,dtype=torch.bfloat16)
        with self.assertRaises(ValueError):kernel.run(m,x,torch.tensor([[0,0]]),out)
        previous=torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with self.assertRaises(ValueError):kernel.run(m,x,torch.tensor([[0,0]]),out)
        finally:torch.set_num_threads(previous)

if __name__=='__main__':unittest.main()
