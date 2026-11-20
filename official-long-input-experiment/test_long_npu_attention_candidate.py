"""Real public-SDK capacity/precision plus ownership and failure ordering."""
import threading,time,types,unittest,math
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import torch
import torch.nn.functional as F
import long_npu_attention_candidate as candidate

def fixture(n=2051,h=2,kh=1):
    generator=torch.Generator().manual_seed(7007)
    def tensor(heads):return (torch.randn((1,n,heads,256),generator=generator)*.25).bfloat16().transpose(1,2)
    return tensor(h),tensor(kh),tensor(kh),torch.ones((1,1,1,n),dtype=torch.bool)

class NpuAttentionTests(unittest.TestCase):
    def setUp(self):
        self.pool=ThreadPoolExecutor(max_workers=2)
        self.owner=candidate.LongNpuAttention(types.SimpleNamespace(engine=types.SimpleNamespace(pool=self.pool)))
    def tearDown(self):
        self.owner.close();self.assertFalse(self.owner.handles)
        self.pool.shutdown(wait=True)
    def test_gqa_padding_and_partial_query_tile_match_fp32_attention(self):
        q,k,v,mask=fixture();mask[...,:7]=False
        with torch.inference_mode():
            actual=self.owner.kernel(q,k,v,mask,1/16)
            expected=F.scaled_dot_product_attention(q.float(),k.float().repeat_interleave(2,1),v.float().repeat_interleave(2,1),attn_mask=mask,dropout_p=0.,is_causal=False,scale=1/16).transpose(1,2).bfloat16()
        delta=actual.float()-expected.float()
        self.assertLessEqual(delta.abs().max().item(),.002)
        self.assertLessEqual((torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(expected.float())).item(),.01)
    def test_last_key_visible_to_all_query_tiles_with_residual_correction(self):
        q,k,v,mask=fixture(h=1);q.zero_();k.zero_();v.zero_();v[:,:,-1]=16;mask[...,:9]=False
        with torch.inference_mode():actual=self.owner.kernel(q,k,v,mask,1/16)
        self.assertTrue(torch.equal(actual,torch.full_like(actual,16/(2051-9))))
    def test_maximum_matrix_capacity_and_partial_rows_use_public_sdk(self):
        handles=[]
        try:
            for k,n in ((256,8192),(8192,256)):
                handle=candidate.lib.la_create(128,k,n,1)
                self.assertTrue(handle,candidate.lib.la_error().decode());handles.append(handle)
                weights=torch.zeros((k,n),dtype=torch.float16)
                self.assertEqual(candidate.lib.la_set_b(handle,weights.data_ptr()),0)
                inputs=torch.ones((1,k),dtype=torch.float16);output=torch.empty((1,n),dtype=torch.float32)
                self.assertEqual(candidate.lib.la_run(handle,inputs.data_ptr(),output.data_ptr(),1),0)
                self.assertTrue(torch.equal(output,torch.zeros_like(output)))
            self.assertFalse(candidate.lib.la_create(128,256,16384,1))
        finally:
            for handle in handles:candidate.lib.la_destroy(handle)
    def test_probability_scaling_preserves_small_contributions_below_fp16_floor(self):
        q,k,v,mask=fixture(n=4097,h=1);q.fill_(1);k.fill_(-.5625);k[:,:,-1]=.5625
        v.fill_(16);v[:,:,-1]=0
        with torch.inference_mode():actual=self.owner.kernel(q,k,v,mask,1/16)
        expected=16*4096*math.exp(-18)/(1+4096*math.exp(-18))
        self.assertTrue(bool((actual>0).all()))
        self.assertLessEqual((actual.float()-expected).abs().max().item(),1e-5)
    def test_partial_creation_cleans_handles_and_keeps_borrowed_pool(self):
        with patch.object(candidate.lib,'la_create',side_effect=[111,222,None]),patch.object(candidate.lib,'la_error',return_value=b'injected create failure'),patch.object(candidate.lib,'la_destroy') as destroy:
            with self.assertRaisesRegex(RuntimeError,'create failure'):self.owner.ensure(2051)
            self.assertEqual([c.args[0] for c in destroy.call_args_list],[111,222])
            self.assertFalse(self.owner.handles)
        self.assertEqual(self.pool.submit(lambda:123).result(),123)
    def test_worker_failure_waits_for_other_worker_before_propagation(self):
        done=threading.Event()
        def worker(index,*args):
            if index==0:raise ValueError('injected first-worker failure')
            time.sleep(.03);done.set()
        with patch.object(self.owner,'ensure'),patch.object(self.owner,'worker',side_effect=worker),torch.inference_mode():
            with self.assertRaisesRegex(ValueError,'first-worker'):self.owner.kernel(*fixture(),1/16)
        self.assertTrue(done.is_set());self.assertFalse(self.owner.active)
    def test_unknown_contract_bypasses_and_runtime_failure_latches_fallback(self):
        module=candidate.qwen.Qwen3_5Attention.__new__(candidate.qwen.Qwen3_5Attention)
        torch.nn.Module.__init__(module);module.training=False;module.scaling=1/16
        old=self.owner.previous;calls=[];sentinel=object()
        def previous(*args,**kwargs):calls.append(kwargs);return sentinel
        self.owner.previous=previous
        try:
            with torch.inference_mode():
                args=fixture()
                self.assertIs(self.owner.forward(module,*args,use_cache=True),sentinel);self.assertTrue(calls[-1]['use_cache'])
                with patch.object(self.owner,'kernel',side_effect=RuntimeError('injected native failure')) as kernel:
                    self.assertIs(self.owner.forward(module,*args),sentinel)
                    self.assertIs(self.owner.forward(module,*args),sentinel);self.assertEqual(kernel.call_count,1)
            self.assertTrue(self.owner.disabled)
        finally:self.owner.previous=old

if __name__=='__main__':unittest.main()
