import types,unittest
import torch
from request_backend_scope_candidate import install

class RequestScopeTests(unittest.TestCase):
    def test_preserves_original_in_request_fallback_and_payload_identity(self):
        seen=[];value=object()
        def original(payload,**kwargs):
            seen.append((payload,kwargs,torch.backends.mkldnn.enabled));torch.backends.mkldnn.enabled=False
            seen.append(torch.backends.mkldnn.enabled);return value
        model=types.SimpleNamespace(system_one=original);owner=types.SimpleNamespace(model=model);close=install(owner);payload=[]
        try:
            with torch.backends.mkldnn.flags(enabled=True):
                self.assertIs(model.system_one(payload,extra=value),value);self.assertTrue(torch.backends.mkldnn.enabled)
            self.assertIs(seen[0][0],payload);self.assertIs(seen[0][1]['extra'],value);self.assertFalse(seen[1])
            self.assertEqual(owner.request_backend_scope_counts,{'calls':1,'restored':1})
        finally:close();close()
        self.assertIs(model.system_one,original)
    def test_exception_and_preceding_false_are_restored(self):
        def original():torch.backends.mkldnn.enabled=not torch.backends.mkldnn.enabled;raise ValueError('injected')
        model=types.SimpleNamespace(system_one=original);owner=types.SimpleNamespace(model=model);close=install(owner)
        try:
            for enabled in (True,False):
                with torch.backends.mkldnn.flags(enabled=enabled):
                    with self.assertRaisesRegex(ValueError,'injected'):model.system_one()
                    self.assertEqual(torch.backends.mkldnn.enabled,enabled)
        finally:close()

if __name__=='__main__':unittest.main()
