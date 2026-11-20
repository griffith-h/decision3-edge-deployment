"""Only the newly reached domain: public SDK at616, protected and B2 bypass."""
import types,unittest
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
import long_npu_attention_candidate as candidate
from test_long_npu_attention_candidate import fixture

class MediumAttentionTests(unittest.TestCase):
    def test_first_new_length_mask_tail_and_full_keys_match_fp32(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            owner=candidate.LongNpuAttention(types.SimpleNamespace(engine=types.SimpleNamespace(pool=pool)),min_length=615)
            try:
                q,k,v,mask=fixture(n=616,h=8,kh=2);mask[...,:3]=False
                with torch.inference_mode():
                    actual=owner.kernel(q,k,v,mask,1/16)
                    expected=F.scaled_dot_product_attention(q.float(),k.float().repeat_interleave(4,1),v.float().repeat_interleave(4,1),attn_mask=mask,dropout_p=0.,is_causal=False,scale=1/16).transpose(1,2).bfloat16()
                delta=actual.float()-expected.float()
                self.assertLessEqual(delta.abs().max().item(),.002)
                self.assertLessEqual((torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(expected.float())).item(),.01)
                self.assertEqual(owner.bucket,640)
            finally:owner.close()
    def test_protected_lengths_and_new_b2_domain_stay_outside_contract(self):
        with torch.inference_mode():
            for n in (115,615):self.assertFalse(candidate.supported(*fixture(n=n),1/16,615))
            args=fixture(n=616)
            self.assertTrue(candidate.supported(*args,1/16,615))
            self.assertFalse(candidate.supported(*args,1/16))
            self.assertFalse(candidate.supported(*(x.repeat(2,1,1,1) for x in args),1/16,615))

if __name__=='__main__':unittest.main()
