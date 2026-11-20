import gc,types,unittest,weakref
import torch
import torch.nn.functional as F
from packed_cache_policy import PackedCachePolicy

def packed():
    module=torch.nn.Linear(64,32,bias=False,dtype=torch.bfloat16);module.weight.requires_grad_(False)
    weight=module.weight.detach().float();bias=None
    module._bp_original=lambda x,w=weight,b=bias:F.linear(x.float(),w,b).bfloat16()
    return module,weakref.ref(weight)

class PackedCacheTests(unittest.TestCase):
    def test_releases_derived_cache_and_private_cpu_keeps_same_output(self):
        module,cache=packed();x=torch.randn(3,64,dtype=torch.bfloat16);expected=module._bp_original(x)
        owner=types.SimpleNamespace(engine=types.SimpleNamespace(packed={1:([module],[32])},pack=lambda ms:([module],[32])))
        policy=PackedCachePolicy(owner);gc.collect();self.assertIsNone(cache())
        self.assertTrue(torch.equal(module._bp_original(x),expected));self.assertEqual(policy.counts['private_cpu_calls'],1)
        self.assertEqual(policy.counts['derived_fp32_bytes_released'],64*32*4);policy.close();policy.close()
    def test_future_pack_retired_once_and_factory_restores(self):
        module,cache=packed();engine=types.SimpleNamespace(packed={},pack=lambda ms:([module],[32]));old=engine.pack
        policy=PackedCachePolicy(types.SimpleNamespace(engine=engine));first=engine.pack([]);self.assertIs(first[0][0],module);engine.pack([])
        self.assertEqual(policy.counts['retired_modules'],1);gc.collect();self.assertIsNone(cache())
        policy.close();self.assertIs(engine.pack,old)
    def test_unknown_cpu_function_not_replaced_and_no_strong_module_cycle(self):
        unknown=torch.nn.Linear(64,32,bias=False,dtype=torch.bfloat16);unknown.eval();unknown._bp_original=unknown.forward;old=unknown._bp_original
        module,cache=packed();engine=types.SimpleNamespace(packed={1:([unknown,module],[32,32])},pack=lambda ms:None)
        policy=PackedCachePolicy(types.SimpleNamespace(engine=engine));self.assertEqual(unknown._bp_original,old);self.assertEqual(policy.counts['guard_bypasses'],1)
        ref=weakref.ref(module);engine.packed.clear();del module;gc.collect();self.assertIsNone(ref());policy.close()
if __name__=='__main__':unittest.main()
