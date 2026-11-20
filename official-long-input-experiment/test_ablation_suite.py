"""Diagnostic switches must restore exactly, including when inference fails."""
import types,unittest
from ablation_suite import perturb

class PerturbationTests(unittest.TestCase):
    def fixture(self):
        layers=[object(),object()];module=object()
        return types.SimpleNamespace(long=types.SimpleNamespace(enabled={'scan','attention','other'}),
            long_mlp=types.SimpleNamespace(disabled={17},states=[(v,) for v in layers]),
            projections=types.SimpleNamespace(disabled={23},modules={'a':module}))
    def test_all_switches_and_failure_restore_original_state_objects(self):
        owner=self.fixture();enabled=owner.long.enabled;mlp=owner.long_mlp.disabled;projection=owner.projections.disabled
        with self.assertRaisesRegex(RuntimeError,'inference failed'):
            with perturb(owner,('scan','attention','mlp','projection')):
                self.assertEqual(owner.long.enabled,{'other'})
                self.assertEqual(len(owner.long_mlp.disabled),3);self.assertEqual(len(owner.projections.disabled),2)
                raise RuntimeError('inference failed')
        self.assertIs(owner.long.enabled,enabled);self.assertIs(owner.long_mlp.disabled,mlp);self.assertIs(owner.projections.disabled,projection)
    def test_unknown_component_cannot_change_state(self):
        owner=self.fixture();original=owner.long.enabled
        with self.assertRaises(ValueError):
            with perturb(owner,('scan','unknown')):pass
        self.assertIs(owner.long.enabled,original)

if __name__=='__main__':unittest.main()
