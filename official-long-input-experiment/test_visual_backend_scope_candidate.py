import types,unittest
import torch
from d3_runtime import _linear_patch_embed_forward
from visual_backend_scope_candidate import install

class VisualScopeTests(unittest.TestCase):
    def fixture(self):
        class Patch(torch.nn.Module):
            def __init__(self):
                super().__init__();self.proj=torch.nn.Conv3d(3,8,(2,2,2),stride=(2,2,2),dtype=torch.bfloat16);self.embed_dim=8
        patch=Patch();patch.forward=types.MethodType(_linear_patch_embed_forward,patch)
        owner=types.SimpleNamespace(model=types.SimpleNamespace(backbone=torch.nn.Sequential(patch)))
        return owner,patch
    def test_real_patch_matches_original_blas_and_restores_true_and_false(self):
        owner,patch=self.fixture();old=patch.forward;x=torch.randn(7,24,dtype=torch.bfloat16)
        with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=False):expected=old(x)
        close=install(owner)
        try:
            for enabled in (True,False):
                with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=enabled):
                    self.assertTrue(torch.equal(patch(x),expected));self.assertEqual(torch.backends.mkldnn.enabled,enabled)
            self.assertEqual(owner.visual_backend_scope_counts,{'calls':2,'scoped':1})
        finally:close();close()
        self.assertEqual(patch.forward,old)
    def test_exception_restores_backend_and_unknown_contract_is_rejected(self):
        owner,patch=self.fixture();old=patch.forward;close=install(owner)
        try:
            with torch.inference_mode(),torch.backends.mkldnn.flags(enabled=True):
                with self.assertRaises(RuntimeError):patch(torch.zeros(1,23,dtype=torch.bfloat16))
                self.assertTrue(torch.backends.mkldnn.enabled)
        finally:close()
        patch.forward=lambda x:x
        with self.assertRaisesRegex(RuntimeError,'exactly one'):install(owner)
        self.assertIsNot(patch.forward,old)

if __name__=='__main__':unittest.main()
