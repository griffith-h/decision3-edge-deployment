"""Boundary and ordering checks, run in the pinned ARM Python environment."""
import types,unittest
import torch
from batch_candidate import BatchPlanner,plan
class BatchTests(unittest.TestCase):
    def test_plan_covers_every_row_once_and_bounds_padded_batches(self):
        lengths=[269,362,263,245,225,225,765,443,7232]
        groups=plan(lengths)
        self.assertEqual(sorted(i for group in groups for i in group),list(range(len(lengths))))
        self.assertTrue(all(len(g)==1 or len(g)*max(lengths[i] for i in g)<=2048 for g in groups))
        self.assertEqual(groups[-1],[8])
    def test_logits_and_option_counts_restore_original_order(self):
        calls=[]
        def original(sequences,counts):
            calls.append(([len(s) for s in sequences],list(counts)))
            return torch.tensor([[s[0],c] for s,c in zip(sequences,counts)],dtype=torch.float32)
        model=types.SimpleNamespace(logits=original);planner=BatchPlanner(model,row_budget=8,max_subbatch=8)
        seq=[[11]*5,[22],[33]*4];counts=[2,7,3]
        out=model.logits(seq,counts)
        self.assertTrue(torch.equal(out,torch.tensor([[11,2],[22,7],[33,3]],dtype=torch.float32)))
        self.assertEqual(len(calls),2);self.assertEqual(counts,[2,7,3]);planner.close();planner.close();self.assertIs(model.logits,original)
    def test_single_long_and_small_batches_delegate_without_changes(self):
        calls=[]
        def original(sequences,counts):calls.append((sequences,counts));return 'original'
        model=types.SimpleNamespace(logits=original);planner=BatchPlanner(model)
        for seq in ([[1]*7232],[[1]*115,[2]*115],[[1]*2303,[2]*2302]):self.assertEqual(model.logits(seq,[2]*len(seq)),'original')
        self.assertEqual(planner.counts['split_batches'],0);self.assertEqual(len(calls),3);planner.close()
    def test_failure_never_returns_partial_batch(self):
        seen=[]
        def original(sequences,counts):
            seen.append(1)
            if len(seen)==2:raise RuntimeError('injected second batch failure')
            return torch.zeros(len(sequences),255)
        model=types.SimpleNamespace(logits=original);planner=BatchPlanner(model,row_budget=4)
        with self.assertRaisesRegex(RuntimeError,'second batch'):model.logits([[1]*4,[2]*3],[2,2])
        planner.close();self.assertIs(model.logits,original)
    def test_disabled_candidate_delegates_oversized_batch(self):
        sentinel=object();model=types.SimpleNamespace(logits=lambda sequences,counts:sentinel)
        planner=BatchPlanner(model);planner.enabled=False
        self.assertIs(model.logits([[1]*765]*8,[2]*8),sentinel)
        self.assertEqual(planner.counts['split_batches'],0);planner.close()
if __name__=='__main__':unittest.main()
