"""Expanded aggregate rows preserve B/T; native size and sequence cap stay bounded."""
import unittest
import torch
from long_projection_candidate import BoundedProjection
from test_long_projection_candidate import fixture

class FullBatchProjectionTests(unittest.TestCase):
    def test_original_batch_shape_preserved_at_new_aggregate_cap(self):
        owner,engine=fixture();scope=BoundedProjection(owner,max_rows=32768)
        try:
            with torch.inference_mode():
                for b,t in ((8,1193),(8,4096)):
                    sequences=[range(t)]*b;counts=[2]*b
                    actual=owner.model.logits(sequences,counts)
                    self.assertEqual(tuple(actual.shape),(b,t,12))
                    self.assertTrue(torch.equal(actual,torch.ones_like(actual)))
                    self.assertEqual([len(s) for s in sequences],[t]*b)
            self.assertTrue(engine.calls);self.assertLessEqual(max(engine.calls),512)
            self.assertEqual(owner.projections.max_rows,2048)
        finally:scope.close();owner.projections.close()
    def test_per_sequence_and_aggregate_caps_still_delegate(self):
        owner,engine=fixture();scope=BoundedProjection(owner,max_rows=32768)
        try:
            with torch.inference_mode():
                for b,t in ((1,8193),(8,4097)):
                    actual=owner.model.logits([range(t)]*b,[2]*b)
                    self.assertTrue(torch.equal(actual,torch.full_like(actual,.5)))
            self.assertFalse(engine.calls);self.assertEqual(scope.counts['scopes'],0)
        finally:scope.close();owner.projections.close()

if __name__=='__main__':unittest.main()
