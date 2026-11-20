import types,unittest
import torch
from medium_dense_candidate import MediumDense

class MediumDenseTests(unittest.TestCase):
    def make(self):
        engine=types.SimpleNamespace(group=lambda *a:None,execute=lambda *a:None)
        native=types.SimpleNamespace(run=lambda *a:None)
        owner=types.SimpleNamespace(engine=engine,native_mlp=native,long_mlp=types.SimpleNamespace(states=[],active=False),long_projection=types.SimpleNamespace(active=False),medium_precision=types.SimpleNamespace(active=False))
        return MediumDense(owner)
    def test_actual_fp32_cpu_forward_tail_and_shape(self):
        torch.manual_seed(34);module=torch.nn.Linear(1024,2048,bias=False,dtype=torch.bfloat16)
        x=torch.randn(2,619,1024,dtype=torch.bfloat16);state=self.make()
        # Repository current CPU forward is cached-FP32 linear, not raw Torch BF16.
        from types import MethodType
        from cpu_optimized_server import linear_forward
        weight=module.weight.detach().float();module.register_buffer('_cpu_fp32_weight',weight);module.register_buffer('_cpu_fp32_bias',None)
        module.forward=MethodType(linear_forward,module)
        with torch.inference_mode():
            expected=module(x);actual=state.linear(module,x)
        self.assertTrue(torch.equal(actual,expected));self.assertEqual(actual.shape,expected.shape)
        state.close();state.close()
    def test_private_output_guard_and_restoration(self):
        state=self.make();state.owner.medium_precision.active=state.owner.long_projection.active=True
        with self.assertRaisesRegex(ValueError,'private'):state.execute(None,None,True)
        with self.assertRaisesRegex(RuntimeError,'active'):state.close()
        state.owner.medium_precision.active=False;state.close()
        self.assertIs(state.engine.execute,state.execute_old)

if __name__=='__main__':unittest.main()
