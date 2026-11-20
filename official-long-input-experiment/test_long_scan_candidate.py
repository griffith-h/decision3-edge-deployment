"""Full-length state carry, FP32 recurrence and native/fallback boundaries."""
import unittest
from unittest.mock import patch
import torch
import long_scan_candidate as candidate

def fixture(n=2053):
    generator=torch.Generator().manual_seed(6006)
    def tensor(d):return (torch.randn((2,2,n,d),generator=generator)*.1).bfloat16().transpose(1,2)
    return tensor(8),tensor(8),tensor(12),torch.full((2,n,2),-.04),torch.full((2,n,2),.15)

def recurrence(q,k,v,g,beta,normalize):
    q,k,v=[x.transpose(1,2).float() for x in (q,k,v)]
    if normalize:
        q=q*torch.rsqrt((q*q).sum(-1,keepdim=True)+1e-6)
        k=k*torch.rsqrt((k*k).sum(-1,keepdim=True)+1e-6)
    q=q*q.shape[-1]**-.5
    state=torch.zeros((q.shape[0],q.shape[1],q.shape[-1],v.shape[-1]));rows=[]
    for t in range(q.shape[2]):
        state=state*g[:,t].exp()[...,None,None]
        old=torch.einsum('bhk,bhkv->bhv',k[:,:,t],state)
        delta=(v[:,:,t]-old)*beta[:,t,:,None]
        state=state+k[:,:,t,:,None]*delta[:,:,None,:]
        rows.append(torch.einsum('bhk,bhkv->bhv',q[:,:,t],state))
    return torch.stack(rows,1).bfloat16()

class ScanTests(unittest.TestCase):
    def test_strided_batches_and_partial_block_match_direct_fp32_state(self):
        args=fixture()
        with torch.inference_mode():
            actual,_=candidate.kernel(*args,True);expected=recurrence(*args,True)
        self.assertLessEqual((actual.float()-expected.float()).abs().max().item(),.002)
    def test_state_from_first_token_reaches_last_token_across_all_blocks(self):
        n=8192;q=torch.zeros((1,n,1,4),dtype=torch.bfloat16);q[...,0]=1
        v=torch.zeros_like(q);v[:,0,:,0]=2;g=torch.zeros((1,n,1));beta=torch.zeros_like(g);beta[:,0]=1
        with torch.inference_mode():actual,_=candidate.kernel(q,q,v,g,beta,False)
        expected=torch.zeros_like(actual);expected[...,0]=1
        self.assertTrue(torch.equal(actual,expected))
    def test_native_capacity_and_coefficient_budget_reject_before_pointer_use(self):
        fn=candidate.lib.d3_chunk_scan
        self.assertEqual(fn(*([None]*6),1,8193,4,4,32,64),-1)
        self.assertEqual(fn(*([None]*6),128,8192,4,4,32,64),-3)
    def test_non_model_gate_domains_do_not_reach_native_state(self):
        q,k,v,g,beta=fixture()
        with torch.inference_mode():
            with self.assertRaises(ValueError):candidate.kernel(q,k,v,g.abs(),beta,True)
            with self.assertRaises(ValueError):candidate.kernel(q,k,v,g,torch.full_like(beta,1.1),True)
            bad=q.clone();bad[:,0,0,0]=float('nan')
            with self.assertRaises(ValueError):candidate.kernel(bad,k,v,g,beta,True)
    def test_cached_state_bypasses_failure_latches_and_function_restores(self):
        calls=[];sentinel=object()
        def previous(*args,**kwargs):calls.append((args,kwargs));return sentinel
        args=fixture();initial=torch.zeros(1)
        with patch.object(candidate.qwen,'torch_chunk_gated_delta_rule',previous):
            owner=candidate.LongScan()
            with torch.inference_mode():
                self.assertIs(owner.forward(*args,initial_state=initial),sentinel)
                self.assertIs(calls[-1][1]['initial_state'],initial)
                with patch.object(candidate,'kernel',side_effect=RuntimeError('injected scan failure')) as kernel:
                    self.assertIs(owner.forward(*args),sentinel);self.assertIs(owner.forward(*args),sentinel)
                    self.assertEqual(kernel.call_count,1)
            owner.close();owner.close();self.assertIs(candidate.qwen.torch_chunk_gated_delta_rule,previous)

if __name__=='__main__':unittest.main()
