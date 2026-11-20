"""Full-key numerical, capacity, fallback and registry checks in the ARM image."""
import unittest
from unittest.mock import patch
import torch
import torch.nn.functional as F
import long_attention_candidate as candidate

class Registry(dict):
    def register(self,key,value):self[key]=value

class AttentionTests(unittest.TestCase):
    def test_strided_gqa_and_mask_match_fp32_reference_above_old_limit(self):
        generator=torch.Generator().manual_seed(2904)
        def tensor(heads):return (torch.randn((1,2049,heads,16),generator=generator)*.25).bfloat16().transpose(1,2)
        q,k,v=tensor(4),tensor(2),tensor(2);mask=torch.ones((1,1,1,2049),dtype=torch.bool);mask[...,0:9]=False
        with torch.inference_mode():
            actual=candidate.kernel(q,k,v,mask,.25)
            expected=F.scaled_dot_product_attention(q.float(),k.float().repeat_interleave(2,1),v.float().repeat_interleave(2,1),attn_mask=mask,dropout_p=0.,is_causal=False,scale=.25).transpose(1,2).bfloat16()
        self.assertLessEqual((actual.float()-expected.float()).abs().max().item(),.001)
    def test_max_length_queries_see_last_key_beyond_every_query_tile(self):
        n=8192;q=torch.zeros((1,1,n,8),dtype=torch.bfloat16);v=torch.zeros_like(q);v[:,:,-1]=16
        mask=torch.ones((1,1,1,n),dtype=torch.bool);mask[...,:16]=False
        with torch.inference_mode():actual=candidate.kernel(q,q,v,mask,1.)
        expected=torch.full_like(actual,16/(n-16));self.assertTrue(torch.equal(actual,expected))
    def test_native_budget_and_length_reject_before_dereferencing_pointers(self):
        fn=candidate.lib.d3_attention_tiled
        self.assertEqual(fn(*([None]*6),1,1,1,8193,8,1.,64,3,1),-1)
        self.assertEqual(fn(*([None]*6),1,64,64,8192,256,1.,64,3,1),-3)
    def test_nonfinite_inputs_are_discarded_and_all_masked_is_unsupported(self):
        q=torch.zeros((1,1,2049,8),dtype=torch.bfloat16);bad=q.clone();bad[...,0,0]=float('nan')
        mask=torch.ones((1,1,1,2049),dtype=torch.bool)
        with torch.inference_mode():
            with self.assertRaises(ValueError):candidate.kernel(bad,q,q,mask,1.)
            self.assertFalse(candidate.supported(q,q,q,torch.zeros_like(mask),1.))
    def test_unknown_contract_bypasses_failure_latches_and_registry_restores(self):
        calls=[];sentinel=object()
        def previous(*args,**kwargs):calls.append((args,kwargs));return sentinel
        registry=Registry(sdpa=previous)
        module=candidate.qwen.Qwen3_5Attention.__new__(candidate.qwen.Qwen3_5Attention)
        torch.nn.Module.__init__(module);module.training=False;module.scaling=1.
        q=torch.zeros((1,1,2049,8),dtype=torch.bfloat16);mask=torch.ones((1,1,1,2049),dtype=torch.bool)
        with patch.object(candidate.qwen,'ALL_ATTENTION_FUNCTIONS',registry):
            owner=candidate.LongAttention()
            with torch.inference_mode():
                self.assertIs(owner.forward(module,q,q,q,mask,is_causal=True),sentinel)
                self.assertIs(calls[-1][0][1],q);self.assertTrue(calls[-1][1]['is_causal'])
                with patch.object(candidate,'kernel',side_effect=ValueError('injected native failure')) as kernel:
                    self.assertIs(owner.forward(module,q,q,q,mask),sentinel)
                    self.assertIs(owner.forward(module,q,q,q,mask),sentinel)
                    self.assertEqual(kernel.call_count,1)
            self.assertTrue(owner.disabled);owner.close();owner.close();self.assertIs(registry['sdpa'],previous)

if __name__=='__main__':unittest.main()
