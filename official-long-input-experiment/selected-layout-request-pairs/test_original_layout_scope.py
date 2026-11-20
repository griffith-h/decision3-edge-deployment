import types,unittest
import torch
from enabled_sequential_repair_candidate import EnabledSequentialRepair
from original_layout_scope import OriginalLayoutScope
from selected_layout_tail import OriginalLayout

class OriginalScopeTests(unittest.TestCase):
    def test_original_kernel_toggle_success_exception_and_restore(self):
        prior=(torch.backends.mkldnn.enabled,torch.get_num_threads(),torch.get_float32_matmul_precision())
        scope=None
        try:
            torch.backends.mkldnn.enabled=True;torch.set_num_threads(3);torch.set_float32_matmul_precision('highest')
            module=types.SimpleNamespace(weight=torch.randn(37,64,dtype=torch.bfloat16))
            x=torch.randn(3,64,dtype=torch.bfloat16);pairs=torch.tensor([[0,0],[1,3],[1,5],[2,36]])
            original=EnabledSequentialRepair();state=types.SimpleNamespace(blas_repair=original)
            owner=types.SimpleNamespace(medium_sparse=types.SimpleNamespace(repair=state),engine=types.SimpleNamespace(packed={}))
            def request(fail=False):
                out=torch.zeros((3,37),dtype=torch.bfloat16);state.blas_repair.run(module,x,pairs,out)
                if fail:raise RuntimeError('requested failure')
                return out
            owner.model=types.SimpleNamespace(system_one=request);expected=request()
            scope=OriginalLayoutScope(owner)
            self.assertIs(type(scope.kernel),OriginalLayout)
            self.assertIs(type(scope.kernel.original),EnabledSequentialRepair)
            for enabled in (False,True,True,False):
                scope.use(enabled);actual=owner.model.system_one()
                self.assertTrue(torch.equal(actual.view(torch.int16),expected.view(torch.int16)))
                self.assertFalse(scope.kernel.active);self.assertEqual(scope.kernel.live_bytes,0);self.assertFalse(scope.kernel.cache)
            self.assertEqual(scope.kernel.counts['builds'],2) # no cross-request reuse
            scope.use(True)
            with self.assertRaisesRegex(RuntimeError,'requested failure'):owner.model.system_one(fail=True)
            self.assertFalse(scope.kernel.active);self.assertEqual(scope.kernel.live_bytes,0);self.assertFalse(scope.kernel.cache)
            scope.kernel.begin()
            with self.assertRaisesRegex(RuntimeError,'Cannot switch'):scope.use(False)
            scope.kernel.end();scope.close();scope.close()
            self.assertIs(state.blas_repair,original);self.assertIs(owner.model.system_one,request)
            with self.assertRaisesRegex(RuntimeError,'Cannot switch'):scope.use(True)
        finally:
            if scope and not scope.closed:scope.close()
            torch.backends.mkldnn.enabled=prior[0];torch.set_num_threads(prior[1]);torch.set_float32_matmul_precision(prior[2])

if __name__=='__main__':unittest.main()
