"""Private consumer scope must restore the public encoder on every exit."""
import types,unittest
from unittest.mock import patch
from long_readout_candidate import LongReadout

def fixture():
    class Encoder:
        def __call__(self,*args):return ['prefix','last']
    encoder=Encoder();model=types.SimpleNamespace(encoder=encoder)
    model.logits=lambda seq,counts:model.encoder(None,None)[-1]
    owner=types.SimpleNamespace(model=model,last_readout=types.SimpleNamespace(
        encoder=encoder,supported=True,disabled=False,closed=False))
    algorithm=LongReadout(owner)
    return model,encoder,algorithm

class ReadoutTests(unittest.TestCase):
    def test_private_single_row_keeps_pooled_output_and_restores_full_encoder(self):
        model,encoder,algorithm=fixture();old=algorithm.original
        with patch.object(algorithm,'good',return_value=True),patch.object(algorithm,'encode',side_effect=lambda *args:encoder(None,None)[-1:]):
            self.assertEqual(model.logits([range(2051)],[2]),'last')
        self.assertIs(model.encoder,encoder);self.assertEqual(model.encoder(None,None),['prefix','last'])
        self.assertEqual(algorithm.counts['optimized'],1);self.assertFalse(algorithm.active)
        algorithm.close();algorithm.close();self.assertIs(model.logits,old)
    def test_failure_restores_encoder_before_replaying_original(self):
        model,encoder,algorithm=fixture()
        with patch.object(algorithm,'good',return_value=True),patch.object(algorithm,'encode',side_effect=ValueError('injected private consumer failure')):
            self.assertEqual(model.logits([range(2051)],[2]),'last')
        self.assertIs(model.encoder,encoder);self.assertTrue(algorithm.disabled)
        self.assertEqual(algorithm.counts['optimized'],0);algorithm.close()
    def test_keyboard_interrupt_also_restores_public_state(self):
        model,encoder,algorithm=fixture()
        with patch.object(algorithm,'good',return_value=True),patch.object(algorithm,'encode',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):model.logits([range(2051)],[2])
        self.assertIs(model.encoder,encoder);self.assertFalse(algorithm.active);algorithm.close()
    def test_original_exception_remains_visible_after_restoration(self):
        model,encoder,algorithm=fixture()
        def fail(seq,counts):raise RuntimeError('original failure')
        algorithm.close();model.logits=fail
        algorithm=LongReadout(types.SimpleNamespace(model=model,last_readout=types.SimpleNamespace(encoder=encoder)))
        with patch.object(algorithm,'good',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'original failure'):model.logits([range(2051)],[2])
        self.assertIs(model.encoder,encoder);self.assertFalse(algorithm.active);algorithm.close()
    def test_short_or_changed_encoder_bypasses_private_consumer(self):
        model,encoder,algorithm=fixture()
        self.assertEqual(model.logits([range(615)],[2]),'last')
        replacement=lambda *args:['new-prefix','new-last'];model.encoder=replacement
        self.assertEqual(model.logits([range(2051)],[2]),'new-last')
        self.assertIs(model.encoder,replacement);self.assertEqual(algorithm.counts['optimized'],0)
        algorithm.close()

if __name__=='__main__':unittest.main()
