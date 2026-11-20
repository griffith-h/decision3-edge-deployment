"""Exercise the existing row tiler through the new scope; no RKNN needed."""
import sys,types,unittest
from pathlib import Path
sys.path.insert(0,'/row')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'row-projection-experiment'))
import torch
from row_projection import RowProjection
from long_projection_candidate import BoundedProjection

class Engine:
    def __init__(self):
        self.groups=[{'parent':'block','members':['q_proj','k_proj','v_proj']}]
        self.cpu_fallbacks=0;self.calls=[];self.fail=None
    def execute(self,m,x):
        self.calls.append(len(x));assert len(x)<=512
        if len(self.calls)==self.fail:raise ValueError('injected tile failure')
        return m._bp_original(x)*2
    def group(self,members,x):return {id(m):self.execute(m,x) for m in members}

def fixture():
    root=torch.nn.Module();root.block=torch.nn.Module();engine=Engine()
    for name in ('q_proj','k_proj','v_proj'):
        m=torch.nn.Linear(4,4,bias=False,dtype=torch.bfloat16);m._bp_original=m.forward
        with torch.no_grad():m.weight.fill_(.125)
        root.block.add_module(name,m)
    root.eval()
    def parent(x):return torch.cat([getattr(root.block,n)(x) for n in ('q_proj','k_proj','v_proj')],-1)
    root.block.forward=parent
    model=types.SimpleNamespace(backbone=root)
    model.logits=lambda seq,counts:root.block(torch.ones(len(seq),max(map(len,seq)),4,dtype=torch.bfloat16))
    owner=types.SimpleNamespace(model=model,fused=True)
    owner.projections=RowProjection(model,engine);owner.projections.mode='chunked'
    return owner,engine

class ProjectionTests(unittest.TestCase):
    def test_long_rows_stay_private_complete_and_native_bounded(self):
        owner,engine=fixture();scope=BoundedProjection(owner)
        try:
            with torch.inference_mode():
                for b,t in ((1,2049),(2,2311),(1,7232),(1,8192)):
                    result=owner.model.logits([range(t)]*b,[2]*b)
                    self.assertEqual(tuple(result.shape),(b,t,12))
                    self.assertTrue(torch.equal(result,torch.ones_like(result)))
            self.assertTrue(engine.calls);self.assertLessEqual(max(engine.calls),512)
            self.assertEqual(owner.projections.max_rows,2048);self.assertTrue(owner.fused)
        finally:scope.close();owner.projections.close()
    def test_oversize_output_budget_and_unknown_length_keep_cpu_path(self):
        owner,engine=fixture();scope=BoundedProjection(owner,max_output_bytes=16)
        try:
            with torch.inference_mode():
                for n in (2049,8193):
                    result=owner.model.logits([range(n)],[2])
                    self.assertTrue(torch.equal(result,torch.full_like(result,.5)))
            self.assertFalse(engine.calls);self.assertGreater(scope.counts['output_budget_fallbacks'],0)
        finally:scope.close();owner.projections.close()
    def test_failed_piece_replays_whole_group_and_restores_scope(self):
        owner,engine=fixture();scope=BoundedProjection(owner);engine.fail=2
        try:
            with torch.inference_mode():result=owner.model.logits([range(2049)],[2])
            self.assertTrue(torch.equal(result,torch.full_like(result,.5)))
            self.assertEqual(owner.projections.counts['range_fallback'],1)
            self.assertEqual(owner.projections.max_rows,2048);self.assertFalse(scope.active)
        finally:scope.close();owner.projections.close()
    def test_short_identity_and_exception_restoration(self):
        owner,engine=fixture();old=owner.model.logits;method=owner.model.backbone.block.q_proj.forward
        scope=BoundedProjection(owner)
        with torch.inference_mode():owner.model.logits([range(615)],[2])
        self.assertEqual(owner.model.backbone.block.q_proj.forward,method);self.assertEqual(scope.counts['scopes'],0)
        scope.close();scope.close();self.assertIs(owner.model.logits,old)
        def fail(seq,counts):raise RuntimeError('model failure')
        owner.model.logits=fail;scope=BoundedProjection(owner)
        try:
            with self.assertRaisesRegex(RuntimeError,'model failure'):owner.model.logits([range(2049)],[2])
            self.assertEqual(owner.projections.max_rows,2048);self.assertTrue(owner.fused);self.assertFalse(scope.active)
        finally:scope.close();owner.projections.close()

if __name__=='__main__':unittest.main()
