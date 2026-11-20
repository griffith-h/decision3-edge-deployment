"""Real-SDK CPU-BF16 reference, complete keys, and changed ownership boundaries."""
import threading,time,types,unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import torch
import torch.nn.functional as F
import medium_flash_candidate as candidate

class MediumFlashTests(unittest.TestCase):
    def setUp(self):
        self.pool=ThreadPoolExecutor(max_workers=2)
        self.owner=candidate.MediumFlash(types.SimpleNamespace(engine=types.SimpleNamespace(pool=self.pool)))
    def tearDown(self):
        self.owner.close();self.owner.close();self.assertFalse(self.owner.handles);self.pool.shutdown(wait=True)
    def fixture(self,n=1193):
        gen=torch.Generator().manual_seed(333)
        def t(h):return (torch.randn((1,n,h,256),generator=gen)*.25).bfloat16().transpose(1,2)
        return t(2),t(1),t(1),torch.ones((1,1,1,n),dtype=torch.bool)
    def test_gqa_tail_and_strides_match_original_bf16_sdpa(self):
        q,k,v,m=self.fixture();m[...,:11]=False
        with torch.inference_mode():
            out=self.owner.kernel(q,k,v,m,1/16)
            ref=F.scaled_dot_product_attention(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),attn_mask=m,dropout_p=0.,is_causal=False,scale=1/16).transpose(1,2)
        delta=out.float()-ref.float()
        self.assertLessEqual(delta.abs().max().item(),.002)
        self.assertLessEqual((torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(ref.float())).item(),.01)
    def test_masked_first_block_and_final_key_visible_to_all_queries(self):
        q,k,v,m=self.fixture();q.zero_();k.zero_();v.zero_();v[:,:,-1]=16;m[...,:512]=False
        with torch.inference_mode():out=self.owner.kernel(q,k,v,m,1/16)
        self.assertTrue(torch.equal(out,torch.full_like(out,16/(1193-512))))
    def test_scope_keeps_protected615_and_rejects_all_masked(self):
        with torch.inference_mode():
            self.assertFalse(candidate.supported(*self.fixture(615),1/16))
            q,k,v,m=self.fixture();m.zero_();self.assertFalse(candidate.supported(q,k,v,m,1/16))
    def test_partial_creation_cleanup_and_borrowed_pool(self):
        with patch.object(candidate.lib,'la_create_flash',side_effect=[111,222,None]),patch.object(candidate.lib,'la_error',return_value=b'injected failure'),patch.object(candidate.lib,'la_destroy') as destroy:
            with self.assertRaisesRegex(RuntimeError,'injected'):self.owner.ensure(1193)
            self.assertEqual([c.args[0] for c in destroy.call_args_list],[111,222]);self.assertFalse(self.owner.handles)
        self.assertEqual(self.pool.submit(lambda:7).result(),7)
    def test_worker_error_joins_before_fallback(self):
        done=threading.Event()
        def worker(i,*args):
            if i==0:raise ValueError('injected worker')
            time.sleep(.03);done.set()
        with patch.object(self.owner,'ensure'),patch.object(self.owner,'worker',side_effect=worker),torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'injected'):self.owner.kernel(*self.fixture(),1/16)
        self.assertTrue(done.is_set());self.assertFalse(self.owner.active)

if __name__=='__main__':unittest.main()
