import unittest
import torch
from sparse_rounding import SparseRepair

class SparseRoundingTests(unittest.TestCase):
    def fixture(self):
        module=torch.nn.Linear(32,4,bias=False,dtype=torch.bfloat16)
        with torch.no_grad():module.weight.zero_();module.weight[:,0]=1
        x=torch.zeros((8,32),dtype=torch.bfloat16);x[:,0]=1
        return module,x
    def test_same_cpu_arithmetic_repairs_private_output_without_mutation(self):
        module,x=self.fixture();raw=torch.full((8,4),1.0039064);saved=raw.clone();before=x.clone()
        repair=SparseRepair(noise_factor=.01,max_fraction=1)
        output=repair.repair(module,x,raw)
        self.assertTrue(torch.equal(output,module(x)));self.assertTrue(torch.equal(raw,saved));self.assertTrue(torch.equal(x,before))
        self.assertEqual(repair.counts['selected'],32);self.assertEqual(repair.counts['changed'],32)
    def test_no_ambiguous_positions_returns_unmodified_rounding(self):
        module,x=self.fixture();repair=SparseRepair();raw=torch.ones(8,4)
        result=repair.repair(module,x,raw)
        self.assertTrue(torch.equal(result,raw.bfloat16()));self.assertEqual(repair.counts['selected'],0)
    def test_partial_final_gemm_tile_is_padded_without_changing_selected_dots(self):
        module,x=self.fixture();raw=torch.ones(8,4);raw.view(-1)[:3]=1.0039064
        repair=SparseRepair(noise_factor=2e-7)
        result=repair.repair(module,x,raw)
        self.assertTrue(torch.equal(result,torch.ones(8,4,dtype=torch.bfloat16)))
        self.assertEqual(repair.counts['selected'],3);self.assertEqual(repair.counts['changed'],3)
        self.assertLessEqual(repair.counts['maximum_gather_float32_bytes'],64*1024*1024)
    def test_versionless_private_weight_requires_explicit_original_guards(self):
        import types
        with torch.inference_mode():weight=torch.ones((4,32),dtype=torch.bfloat16)
        module=types.SimpleNamespace(weight=weight,bias=None,out_features=4)
        x=torch.ones((8,32),dtype=torch.bfloat16);raw=torch.full((8,4),32.)
        repair=SparseRepair()
        with self.assertRaisesRegex(ValueError,'original-parameter guards'):repair.repair(module,x,raw)
        result=repair.repair(module,x,raw,allow_inference_weight=True)
        self.assertTrue(torch.equal(result,raw.bfloat16()));self.assertIsNone(repair.cache[id(module)][1])
    def test_density_failure_preserves_original_inputs(self):
        module,x=self.fixture();raw=torch.full((8,4),1.0039064);saved=raw.clone()
        with self.assertRaisesRegex(ValueError,'density budget'):SparseRepair(noise_factor=.01).repair(module,x,raw)
        self.assertTrue(torch.equal(raw,saved))
    def test_weight_version_invalidates_only_cached_norms(self):
        module,x=self.fixture();repair=SparseRepair(noise_factor=.01,max_fraction=1)
        repair.repair(module,x,torch.ones(8,4));first=repair.cache[id(module)]
        with torch.no_grad():module.weight.mul_(2)
        repair.repair(module,x,torch.full((8,4),2.));second=repair.cache[id(module)]
        self.assertNotEqual(first[1],second[1]);self.assertEqual(repair.counts['weight_norm_builds'],2)
    def test_nonfinite_or_wrong_layout_does_not_reach_repair(self):
        module,x=self.fixture();repair=SparseRepair()
        with self.assertRaisesRegex(ValueError,'Nonfinite'):repair.repair(module,x,torch.full((8,4),float('nan')))
        with self.assertRaisesRegex(ValueError,'Unsupported'):repair.repair(module,x.float(),torch.ones(8,4))
        self.assertEqual(repair.counts['calls'],0)

if __name__=='__main__':unittest.main()
