import types,unittest
from types import MethodType
import torch
from medium_cpu_original import MediumCpuOriginal

class Qwen3_5RMSNorm(torch.nn.Module):
    def forward(self,x):return x+1

def fixture(fail=False):
    norm=Qwen3_5RMSNorm().eval();reference=norm.forward
    def native(self,x,_old=reference):return x+2
    native.__module__='lut_ops';norm.forward=MethodType(native,norm)
    backbone=torch.nn.Module();backbone.add_module('norm',norm);backbone.eval()
    model=types.SimpleNamespace(backbone=backbone)
    def logits(self,sequences,counts):
        self.last_seen=(sequences,counts)
        if fail:raise RuntimeError('injected model failure')
        return norm(torch.ones(1))
    model.logits=MethodType(logits,model)
    return types.SimpleNamespace(model=model),norm

class OriginalCpuScopeTests(unittest.TestCase):
    def test_same_sequence_objects_cpu_method_and_exact_restoration(self):
        owner,norm=fixture();old=norm.forward;state=MediumCpuOriginal(owner)
        with torch.inference_mode():
            sequences=[[0]*1193]*2;counts=[1,1]
            value=owner.model.logits(sequences,counts);self.assertEqual(value.item(),2)
            self.assertIs(owner.model.last_seen[0],sequences);self.assertIs(owner.model.last_seen[1],counts)
            self.assertIs(norm.forward,old)
        state.close();state.close()
    def test_protected_and_long_domains_keep_native_method(self):
        owner,norm=fixture();state=MediumCpuOriginal(owner)
        with torch.inference_mode():
            for b,t in ((1,115),(1,615),(1,7232),(3,239)):
                self.assertEqual(owner.model.logits([[0]*t]*b,[1]*b).item(),3)
        self.assertEqual(state.counts['scopes'],0);state.close()
    def test_exception_and_foreign_method_guard_preserve_exact_objects(self):
        owner,norm=fixture(True);old=norm.forward;state=MediumCpuOriginal(owner)
        with torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError,'injected'):owner.model.logits([[0]*1193]*2,[1,1])
        self.assertFalse(state.active);self.assertIs(norm.forward,old);state.close()
        owner,norm=fixture();state=MediumCpuOriginal(owner);foreign=lambda x:x+7;norm.forward=foreign
        with torch.inference_mode():self.assertEqual(owner.model.logits([[0]*1193]*2,[1,1]).item(),8)
        self.assertEqual(state.counts['guard_bypass'],1);self.assertIs(norm.forward,foreign);state.close()
if __name__=='__main__':unittest.main()
