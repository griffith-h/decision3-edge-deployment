"""Run inside the pinned ARM image; no NPU or model weights needed for these tests."""
import types
import unittest
import torch
from long_mlp_candidate import BoundedMlp

class Qwen3_5MLP(torch.nn.Module):
    def __init__(self):
        super().__init__();self.gate_proj=torch.nn.Linear(1024,4,bias=False);self.up_proj=torch.nn.Linear(1024,4,bias=False);self.down_proj=torch.nn.Linear(4,1024,bias=False);self.act_fn=torch.nn.SiLU();self.eval();self.to(torch.bfloat16)
    def forward(self,x):return x*3
class FakeNative:
    def __init__(self,fail_at=None):self.calls=0;self.fail_at=fail_at;self.tail=None
    def run(self,members,x):
        self.calls+=1;assert tuple(x.shape)==(1,615,1024)
        if self.calls==self.fail_at:raise ValueError('injected chunk range failure')
        self.tail=x.clone();return x*2
class BoundedTests(unittest.TestCase):
    def owner(self,shape=(2,1100,1024),fail_at=None):
        layer=Qwen3_5MLP();x=(torch.arange(torch.tensor(shape).prod().item()).reshape(shape)%17).to(torch.bfloat16)
        model=types.SimpleNamespace(backbone=torch.nn.ModuleList([layer]),logits=lambda sequences,counts:layer(x));native=FakeNative(fail_at)
        owner=types.SimpleNamespace(model=model,native_mlp=native);return owner,layer,x
    def test_preserves_all_batch_rows_and_pads_only_private_tail(self):
        owner,layer,x=self.owner();old=layer.forward;scope=BoundedMlp(owner)
        with torch.inference_mode():actual=owner.model.logits([range(1100)]*2,[2,2])
        self.assertTrue(torch.equal(actual,x*2));self.assertEqual(scope.counts['chunks'],4)
        self.assertTrue(torch.equal(owner.native_mlp.tail[:,355:],torch.zeros_like(owner.native_mlp.tail[:,355:])))
        self.assertEqual(layer.forward,old);scope.close();scope.close()
    def test_failure_discards_prior_chunks_and_replays_complete_old_mlp(self):
        owner,layer,x=self.owner(fail_at=2);old=layer.forward;scope=BoundedMlp(owner)
        with torch.inference_mode():actual=owner.model.logits([range(1100)]*2,[2,2])
        self.assertTrue(torch.equal(actual,x*3));self.assertEqual(scope.counts['layers'],0);self.assertEqual(scope.counts['range_fallbacks'],1)
        self.assertEqual(layer.forward,old);self.assertFalse(scope.active);scope.close()
    def test_runtime_failure_replays_whole_layer_and_latches_cpu(self):
        owner,layer,x=self.owner();scope=BoundedMlp(owner)
        def broken(members,part):raise RuntimeError('injected native runtime failure')
        owner.native_mlp.run=broken
        with torch.inference_mode():
            first=owner.model.logits([range(1100)]*2,[2,2])
            second=owner.model.logits([range(1100)]*2,[2,2])
        self.assertTrue(torch.equal(first,x*3));self.assertTrue(torch.equal(second,x*3))
        self.assertEqual(scope.counts['runtime_errors'],['injected native runtime failure'])
        self.assertEqual(scope.counts['guard_fallbacks'],1);scope.close()
    def test_615_shape_keeps_original_forward_identity(self):
        owner,layer,x=self.owner((1,615,1024));old=layer.forward;scope=BoundedMlp(owner)
        with torch.inference_mode():actual=owner.model.logits([range(615)],[2])
        self.assertEqual(owner.native_mlp.calls,0);self.assertEqual(layer.forward,old);self.assertTrue(torch.equal(actual,x*3));scope.close()
    def test_changed_weight_is_not_accelerated(self):
        owner,layer,x=self.owner();scope=BoundedMlp(owner)
        with torch.no_grad():layer.gate_proj.weight.add_(1)
        with torch.inference_mode():actual=owner.model.logits([range(1100)]*2,[2,2])
        self.assertTrue(torch.equal(actual,x*3));self.assertEqual(owner.native_mlp.calls,0);scope.close()
    def test_forward_restored_if_model_raises(self):
        owner,layer,x=self.owner();old=layer.forward
        def fail(sequences,counts):raise RuntimeError('injected model failure')
        owner.model.logits=fail;scope=BoundedMlp(owner)
        with self.assertRaises(RuntimeError):owner.model.logits([range(1100)]*2,[2,2])
        self.assertEqual(layer.forward,old);self.assertFalse(scope.active);scope.close();self.assertIs(owner.model.logits,fail)
if __name__=='__main__':unittest.main()
