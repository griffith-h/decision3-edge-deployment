import types,unittest
from unittest.mock import patch
import torch
from enabled_sequential_repair_candidate import EnabledSequentialRepair
from selected_dot_ilp import SelectedDot
from selected_layout_tail import LayoutTail,OriginalLayout,install

class LayoutTailTests(unittest.TestCase):
    def setUp(self):
        self.previous=(torch.backends.mkldnn.enabled,torch.get_num_threads(),torch.get_float32_matmul_precision())
        torch.backends.mkldnn.enabled=True;torch.set_num_threads(3);torch.set_float32_matmul_precision('highest')
    def tearDown(self):
        torch.backends.mkldnn.enabled=self.previous[0];torch.set_num_threads(self.previous[1]);torch.set_float32_matmul_precision(self.previous[2])
    def test_strides_all_tails_bf16_bits_and_fp32_k_order(self):
        g=torch.Generator().manual_seed(91);old=EnabledSequentialRepair();ref=SelectedDot(4)
        with torch.inference_mode():
            for k in (4,64,1024,2048,3584,6144,10240):
                x=torch.randn(9,k+8,dtype=torch.bfloat16,generator=g)[:,:k]
                m=types.SimpleNamespace(weight=torch.randn(41,k+8,dtype=torch.bfloat16,generator=g)[:,:k])
                pairs=torch.tensor([[r,c] for r,count in enumerate((1,3,4,7,8,15,16,17,31)) for c in range(count)],dtype=torch.long)
                expected=torch.full((9,41),77.,dtype=torch.bfloat16);old.run(m,x,pairs,expected)
                raw=torch.empty(len(pairs),dtype=torch.float32);ref.run(m,x,pairs,expected,raw)
                for mode,pad in ((4,True),(16,False),(16,True)):
                    kernel=LayoutTail(mode,pad,trusted_inference=lambda _:True);kernel.begin()
                    try:
                        out=torch.full_like(expected,77.);debug=torch.empty_like(raw);kernel.run(m,x,pairs,out,debug)
                        self.assertTrue(torch.equal(out.view(torch.int16),expected.view(torch.int16)),(k,mode,pad))
                        self.assertTrue(torch.equal(debug.view(torch.int32),raw.view(torch.int32)),(k,mode,pad))
                    finally:kernel.close()
                    self.assertEqual(kernel.live_bytes,0)
    def test_cache_reuse_mutation_replacement_eviction_and_oversize(self):
        m=types.SimpleNamespace(weight=torch.ones(32,64,dtype=torch.bfloat16));other=types.SimpleNamespace(weight=torch.full((32,64),2.,dtype=torch.bfloat16));replacement=torch.full_like(m.weight,3.);giant=types.SimpleNamespace(weight=torch.ones(64,64,dtype=torch.bfloat16))
        kernel=LayoutTail(max_bytes=7000);kernel.begin()
        with torch.inference_mode():
            first=kernel.layout(m);self.assertIs(first,kernel.layout(m));self.assertEqual(kernel.counts['builds'],1)
            m.weight.add_(1);second=kernel.layout(m);self.assertIsNot(first,second);self.assertTrue(bool((second==2).all()))
            m.weight=replacement;self.assertTrue(bool((kernel.layout(m)==3).all()))
            kernel.layout(other);self.assertEqual(kernel.counts['evictions'],1);self.assertLessEqual(kernel.live_bytes,7000)
            self.assertIs(kernel.layout(giant),giant.weight)
        kernel.end();self.assertEqual(kernel.live_bytes,0);self.assertEqual(len(kernel.cache),0);kernel.close()
    def test_versionless_guard_backend_scope_and_invalid_positions(self):
        with torch.inference_mode():m=types.SimpleNamespace(weight=torch.ones(32,64,dtype=torch.bfloat16))
        x=torch.ones(3,64,dtype=torch.bfloat16);out=torch.full((3,32),77.,dtype=torch.bfloat16);kernel=LayoutTail();kernel.begin()
        with self.assertRaisesRegex(ValueError,'Versionless'):kernel.run(m,x,torch.tensor([[0,0]]),out)
        self.assertTrue(bool((out==77).all()));kernel.close()
        m=types.SimpleNamespace(weight=torch.ones(32,64,dtype=torch.bfloat16));kernel=LayoutTail();kernel.begin()
        for pairs in (torch.tensor([[2,0],[0,0]]),torch.tensor([[3,0]]),torch.tensor([[0,32]])):
            with self.assertRaises(ValueError):kernel.run(m,x,pairs,out)
            self.assertTrue(bool((out==77).all()))
        kernel.run(m,x,torch.empty((0,2),dtype=torch.long),out);self.assertTrue(bool((out==77).all()))
        kernel.run(m,x,torch.tensor([[0,2],[0,2],[2,31]]),out);self.assertEqual(float(out[0,2]),64.)
        torch.backends.mkldnn.enabled=False
        with self.assertRaisesRegex(ValueError,'backend changed'):kernel.run(m,x,torch.tensor([[0,0]]),out)
        kernel.close()
    def test_original_kernel_layout_control_preserves_original_bits(self):
        m=types.SimpleNamespace(weight=torch.randn(37,2048,dtype=torch.bfloat16));x=torch.randn(3,2048,dtype=torch.bfloat16);pairs=torch.tensor([[r,c] for r in range(3) for c in range(17)])
        expected=torch.full((3,37),77.,dtype=torch.bfloat16);EnabledSequentialRepair().run(m,x,pairs,expected)
        kernel=OriginalLayout();kernel.begin();out=torch.full_like(expected,77.);raw=torch.empty(len(pairs),dtype=torch.float32)
        kernel.run(m,x,pairs,out,raw);self.assertTrue(torch.equal(out.view(torch.int16),expected.view(torch.int16)))
        kernel.run(m,x,pairs,out);self.assertEqual(kernel.counts['builds'],1);self.assertEqual(kernel.counts['hits'],1);kernel.close();self.assertEqual(kernel.live_bytes,0)
    def test_request_exception_clears_cache_and_restores_owner(self):
        module=types.SimpleNamespace(weight=torch.ones(32,64,dtype=torch.bfloat16));state=types.SimpleNamespace(blas_repair=object());owner=types.SimpleNamespace(medium_sparse=types.SimpleNamespace(repair=state),engine=types.SimpleNamespace(packed={}))
        def request(fail=False):
            owner.selected_layout.kernel.layout(module)
            if fail:raise RuntimeError('requested failure')
            return 7
        owner.model=types.SimpleNamespace(system_one=request);original=state.blas_repair;close=install(owner)
        self.assertEqual(owner.model.system_one(),7);self.assertEqual(owner.selected_layout.kernel.live_bytes,0)
        with self.assertRaisesRegex(RuntimeError,'requested failure'):owner.model.system_one(fail=True)
        self.assertFalse(owner.selected_layout.kernel.active);self.assertEqual(owner.selected_layout.kernel.live_bytes,0)
        close();close();self.assertIs(owner.model.system_one,request);self.assertIs(state.blas_repair,original)

if __name__=='__main__':unittest.main()
