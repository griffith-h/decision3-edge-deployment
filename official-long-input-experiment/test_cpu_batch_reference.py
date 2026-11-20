"""Reference grouping/formatting and accidental acceleration guards."""
import json,tempfile,types,unittest
from pathlib import Path
from cpu_batch_reference import run

class CpuBatchReferenceTests(unittest.TestCase):
    def fixture(self,accelerate=False,change_format=False):
        owner=types.SimpleNamespace(engine=types.SimpleNamespace(calls=7),
            native_mlp=types.SimpleNamespace(calls=11),npu_calls=5,
            bank=types.SimpleNamespace(counts={'graph_calls':0}))
        lengths=[20,1,2,3,4,5,6,7,30,8,9,10,11,12,13,14,40,15]
        questions={f'q{i:02d}':{'length':n} for i,n in enumerate(lengths)};calls=[]
        model=types.SimpleNamespace(batch_size=8,max_length=None,model_name='fixed')
        def prepare(**payload):
            q=payload['questions'];extra=int(change_format and len(q)<18)
            return types.SimpleNamespace(runnable=list(q),sequences={k:list(range(v['length']+extra)) for k,v in q.items()},errors={})
        def system_one(**payload):
            q=payload['questions'];calls.append(list(q));width=max(v['length'] for v in q.values())
            if accelerate:owner.engine.calls+=1
            return {'model':'fixed','answers':{k:{'choice':str(width)} for k in q},
                    'usage':{'input_tokens':sum(v['length'] for v in q.values()),'output_tokens':0}}
        model.prepare=prepare;model.system_one=system_one
        return model,owner,{'state':'synthetic','questions':questions},calls
    def test_original8_8_2_padding_and_order_kept(self):
        model,owner,request,calls=self.fixture()
        with tempfile.TemporaryDirectory() as path:
            run(model,owner,request,lambda event:None,Path(path),'case')
            data=json.loads((Path(path)/'result.json').read_text())
        self.assertEqual([len(c) for c in calls],[8,8,2]);self.assertEqual(sum(calls,[]),list(request['questions']))
        self.assertEqual([r['width'] for r in data['records']],[20,30,40])
        self.assertEqual(data['response']['answers']['q01']['choice'],'20')
        self.assertTrue(data['completed']);self.assertEqual(data['npu_matrix_calls'],0)
    def test_accidental_npu_use_never_advertises_completed_reference(self):
        model,owner,request,calls=self.fixture(accelerate=True)
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(AssertionError,'unexpectedly used NPU'):
                run(model,owner,request,lambda event:None,Path(path),'case')
            data=json.loads((Path(path)/'result.json').read_text())
            self.assertFalse(data['completed']);self.assertEqual(len(data['records']),1)
    def test_changed_prompt_tokens_rejected_before_inference(self):
        model,owner,request,calls=self.fixture(change_format=True)
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(AssertionError):run(model,owner,request,lambda event:None,Path(path),'case')
        self.assertFalse(calls)

if __name__=='__main__':unittest.main()
