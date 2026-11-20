import types,unittest
import torch
from medium_sparse_candidate import MediumSparse

class MediumSparseTests(unittest.TestCase):
    def fixture(self):
        result=object();engine=types.SimpleNamespace(cpu_fallbacks=0,ensure=lambda m:17)
        engine.group=lambda members,x:result
        engine.execute=lambda module,x,prepared=False,output=None:result
        native=types.SimpleNamespace(run=lambda members,x:result,handles={})
        owner=types.SimpleNamespace(engine=engine,native_mlp=native,
            long_mlp=types.SimpleNamespace(states=[],active=True),long_projection=types.SimpleNamespace(active=True),
            medium_precision=types.SimpleNamespace(active=False))
        return owner,result
    def test_inactive_passes_through_exact_objects_and_restores_idempotently(self):
        owner,result=self.fixture();original=(owner.engine.group,owner.engine.execute,owner.native_mlp.run)
        adapter=MediumSparse(owner)
        try:
            self.assertIs(owner.engine.group([],None),result);self.assertIs(owner.engine.execute(None,None),result)
            self.assertIs(owner.native_mlp.run([],None),result)
        finally:adapter.close();adapter.close()
        self.assertEqual((owner.engine.group,owner.engine.execute,owner.native_mlp.run),original)
    def test_whole_cpu_fallback_is_not_overwritten_by_repair(self):
        owner,result=self.fixture();owner.medium_precision.active=True
        def fallback(members,x):owner.engine.cpu_fallbacks+=1;return result
        owner.engine.group=fallback;adapter=MediumSparse(owner)
        try:self.assertIs(owner.engine.group([],None),result)
        finally:adapter.close()
        self.assertFalse(adapter.group_active)
    def test_failed_group_restores_active_flag_before_propagation(self):
        owner,_=self.fixture();owner.medium_precision.active=True
        def fail(members,x):raise RuntimeError('Worker failed after join')
        owner.engine.group=fail;adapter=MediumSparse(owner)
        try:
            with self.assertRaisesRegex(RuntimeError,'Worker failed'):owner.engine.group([],None)
            self.assertFalse(adapter.group_active)
        finally:adapter.close()
    def test_invalid_fp32_context_is_rejected_before_returning_private_result(self):
        owner,_=self.fixture();adapter=MediumSparse(owner)
        adapter.norm=types.SimpleNamespace(d3_projection_read_fp32=lambda *args:-1)
        try:
            with self.assertRaisesRegex(ValueError,'context read failed'):
                adapter.correct_norm(torch.nn.Linear(32,4,bias=False,dtype=torch.bfloat16),torch.zeros(3,32,dtype=torch.bfloat16))
            self.assertEqual(adapter.counts['read_errors'],1)
        finally:adapter.close()

if __name__=='__main__':unittest.main()
