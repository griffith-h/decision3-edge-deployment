import types,unittest
from medium_precision_candidate import MediumPrecision
from medium_scan_restore_candidate import install

class MediumScanPolicyTests(unittest.TestCase):
    def fixture(self,fail=False):
        owner=types.SimpleNamespace(long=types.SimpleNamespace(enabled={'scan','attention','other'}));seen=[]
        def logits(sequences,counts):
            seen.append(owner.long.enabled)
            if fail:raise ValueError('injected inference failure')
            return sequences,counts
        owner.model=types.SimpleNamespace(logits=logits);owner.medium_precision=MediumPrecision(owner)
        return owner,seen
    def test_native_scan_only_in_original_guarded_medium_scope_and_exact_restore(self):
        owner,seen=self.fixture();enabled=owner.long.enabled;close=install(owner)
        seq=[[1]*1193,[2]*1102];counts=[2,3];returned=owner.model.logits(seq,counts)
        self.assertIs(returned[0],seq);self.assertIs(returned[1],counts)
        self.assertEqual(seen[-1],{'scan','other'});self.assertIs(owner.long.enabled,enabled)
        self.assertEqual(owner.medium_precision.counts['native_scan_scopes'],1)
        for batch,length in ((1,115),(1,615),(1,2049),(9,1193)):
            owner.model.logits([[1]*length for _ in range(batch)],[2]*batch)
            self.assertIs(seen[-1],enabled)
        close();close();self.assertFalse(owner.medium_precision.native_scan_enabled)
        owner.model.logits(seq,counts);self.assertEqual(seen[-1],{'other'})
        owner.medium_precision.close()
    def test_failure_restores_enabled_state_and_active_policy_cannot_be_closed(self):
        owner,_=self.fixture(True);enabled=owner.long.enabled;close=install(owner)
        with self.assertRaisesRegex(ValueError,'injected'):
            owner.model.logits([[1]*1193],[2])
        self.assertIs(owner.long.enabled,enabled);self.assertFalse(owner.medium_precision.active)
        owner.medium_precision.active=True
        with self.assertRaisesRegex(RuntimeError,'active precision scope'):close()
        self.assertTrue(owner.medium_precision.native_scan_enabled)
        owner.medium_precision.active=False;close();self.assertFalse(owner.medium_precision.native_scan_enabled)
        owner.medium_precision.close()

if __name__=='__main__':unittest.main()
