import types,unittest
from medium_precision_candidate import MediumPrecision

class MediumPrecisionTests(unittest.TestCase):
    def fixture(self,fail=False):
        owner=types.SimpleNamespace(long=types.SimpleNamespace(enabled={'scan','attention','other'}));observed=[]
        def logits(sequences,counts):
            observed.append(owner.long.enabled)
            if fail:raise RuntimeError('Inference failed')
            return sequences,counts
        owner.model=types.SimpleNamespace(logits=logits);return owner,observed
    def test_original_batches_unchanged_and_original_state_restored(self):
        owner,observed=self.fixture();enabled=owner.long.enabled;original=owner.model.logits
        state=MediumPrecision(owner);sequences=[[1]*1193,[2]*1102];counts=[2,3]
        returned=owner.model.logits(sequences,counts)
        self.assertIs(returned[0],sequences);self.assertIs(returned[1],counts)
        self.assertEqual(observed,[{'other'}]);self.assertIs(owner.long.enabled,enabled)
        state.close();state.close();self.assertIs(owner.model.logits,original)
    def test_protected_and_long_domains_unchanged(self):
        for batch,length in ((1,115),(1,512),(1,615),(1,2049),(1,8192),(9,1193)):
            owner,observed=self.fixture();enabled=owner.long.enabled;state=MediumPrecision(owner)
            try:owner.model.logits([[1]*length for _ in range(batch)],[2]*batch)
            finally:state.close()
            self.assertEqual(state.counts['scopes'],0);self.assertIs(observed[0],enabled)
    def test_failure_restores_original_state(self):
        owner,_=self.fixture(True);enabled=owner.long.enabled;state=MediumPrecision(owner)
        with self.assertRaisesRegex(RuntimeError,'Inference failed'):owner.model.logits([[1]*1193],[2])
        self.assertIs(owner.long.enabled,enabled);self.assertFalse(state.active);state.close()

if __name__=='__main__':unittest.main()
