import types,unittest
import torch
from scan_threads_candidate import install

class ScanThreadTests(unittest.TestCase):
    def test_only_medium_scope_changes_threads_and_restores_original_callable(self):
        saved=torch.get_num_threads();torch.set_num_threads(3);seen=[];payload=object()
        def original(x):seen.append((x,torch.get_num_threads()));return x
        owner=types.SimpleNamespace(long=types.SimpleNamespace(original_scan=original),medium_precision=types.SimpleNamespace(active=True));close=install(owner)
        try:
            self.assertIs(owner.long.original_scan(payload),payload);self.assertEqual(seen[-1],(payload,1));self.assertEqual(torch.get_num_threads(),3)
            owner.scan_threads.enabled=False;owner.long.original_scan(payload);self.assertEqual(seen[-1][1],3)
            owner.scan_threads.enabled=True;owner.medium_precision.active=False;owner.long.original_scan(payload);self.assertEqual(seen[-1][1],3)
            self.assertEqual(owner.scan_threads.counts['scopes'],1)
        finally:close();close();torch.set_num_threads(saved)
        self.assertIs(owner.long.original_scan,original)
    def test_exception_and_wrong_entry_threads_cannot_leak(self):
        saved=torch.get_num_threads();torch.set_num_threads(3)
        def original():raise ValueError('injected')
        owner=types.SimpleNamespace(long=types.SimpleNamespace(original_scan=original),medium_precision=types.SimpleNamespace(active=True));close=install(owner)
        try:
            with self.assertRaisesRegex(ValueError,'injected'):owner.long.original_scan()
            self.assertEqual(torch.get_num_threads(),3);self.assertFalse(owner.scan_threads.active)
            torch.set_num_threads(1)
            with self.assertRaisesRegex(RuntimeError,'Unexpected original'):owner.long.original_scan()
        finally:close();torch.set_num_threads(saved)

if __name__=='__main__':unittest.main()
