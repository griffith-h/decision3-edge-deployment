"""Observation must preserve returned objects, restore globals, and expose errors."""
import tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import torch
import long_kernels as lk
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from component_observer import Observer,ObservationError

class ObservationTests(unittest.TestCase):
    def fixture(self):
        module=types.SimpleNamespace(_bp_original=lambda x:x)
        owner=types.SimpleNamespace(projections=types.SimpleNamespace(run=lambda *a:'projection',modules={'p':module}),
            native_mlp=types.SimpleNamespace(run=lambda *a:'mlp'),long_mlp=types.SimpleNamespace(states=[]),
            long=types.SimpleNamespace(original_scan=lambda *a,**kw:('scan',None)))
        return owner,module
    def test_passthrough_identity_and_idempotent_restore(self):
        owner,_=self.fixture();p=owner.projections.run;m=owner.native_mlp.run
        a=lk.attention;s=qwen.torch_chunk_gated_delta_rule
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(lk,'attention',return_value='attention'),patch.object(qwen,'torch_chunk_gated_delta_rule',return_value=('scan',None)):
                attention=lk.attention;scan=qwen.torch_chunk_gated_delta_rule
                observer=Observer(owner,Path(folder))
                self.assertEqual(owner.projections.run([],None,None),'projection');self.assertEqual(owner.native_mlp.run([],None),'mlp')
                self.assertEqual(lk.attention(None,None,None,None,None),'attention');self.assertEqual(qwen.torch_chunk_gated_delta_rule(None),('scan',None))
                observer.close();observer.close()
                self.assertIs(owner.projections.run,p);self.assertIs(owner.native_mlp.run,m)
                self.assertIs(lk.attention,attention);self.assertIs(qwen.torch_chunk_gated_delta_rule,scan)
        self.assertIs(lk.attention,a);self.assertIs(qwen.torch_chunk_gated_delta_rule,s)
    def test_observation_error_is_not_a_fallback_runtime_or_value_error(self):
        owner,module=self.fixture();returned={id(module):torch.ones(1)};owner.projections.run=lambda *a:returned
        def broken(x):raise RuntimeError('Reference failed')
        module._bp_original=broken
        with tempfile.TemporaryDirectory() as folder:
            observer=Observer(owner,Path(folder));observer.mode='only_projection'
            try:
                with self.assertRaises(ObservationError) as result:owner.projections.run([module],torch.ones(1),None)
                self.assertNotIsInstance(result.exception,(RuntimeError,ValueError));self.assertEqual(observer.done,set())
            finally:observer.close()
    def test_statistics_do_not_mutate_inputs(self):
        owner,_=self.fixture();a=torch.tensor([1.,1.0078125,-1.,0.],dtype=torch.bfloat16);b=torch.tensor([1.,1.,-1.,0.],dtype=torch.bfloat16)
        before=a.clone()
        with tempfile.TemporaryDirectory() as folder:
            observer=Observer(owner,Path(folder))
            try:
                observer.record('example',a,b,{})
                self.assertTrue(torch.equal(a,before));row=observer.records[0]
                self.assertEqual(row['same_sign_nonzero_ulp_counts'],{'0':2,'1':1,'2':0})
            finally:observer.close()

if __name__=='__main__':unittest.main()
