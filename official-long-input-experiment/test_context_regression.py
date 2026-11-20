import json,tempfile,types,unittest
from pathlib import Path
from context_regression import run,EXPECTED,require_backend

class ContextRegressionTests(unittest.TestCase):
    def fixture(self,backend_bad=False,npu=False):
        state=dict(EXPECTED);owner=types.SimpleNamespace(request_backend_scope_counts={'calls':0,'restored':0});calls=[]
        def system_one(**payload):
            calls.append(payload);owner.request_backend_scope_counts['calls']+=1
            if backend_bad:state['mkldnn']=False
            return {'answers':{},'usage':{'input_tokens':1}}
        model=types.SimpleNamespace(system_one=system_one)
        rows=[{'id':str(i),'payload':{'state':str(i),'questions':{}}} for i in range(34)]
        def counters(owner):return (len(calls) if npu else 0,0,0,0)
        return model,owner,rows,lambda stage:{**state,'stage':stage},counters,calls
    def test_all34_preserve_payload_order_and_record_context_and_zero_cpu_npu(self):
        model,owner,rows,backend,counters,calls=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            run(model,owner,rows,lambda:{},backend,lambda e:None,Path(folder),'reference',b'fixed',counters)
            data=json.loads((Path(folder)/'context-regression.json').read_text())
        self.assertTrue(data['completed']);self.assertEqual(len(data['records']),34)
        self.assertEqual(calls,[r['payload'] for r in rows]);self.assertEqual(data['case_ids'],[r['id'] for r in rows])
    def test_context_drift_and_unexpected_cpu_npu_are_preserved_and_rejected(self):
        for bad,npu in ((True,False),(False,True)):
            model,owner,rows,backend,counters,_=self.fixture(bad,npu)
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(RuntimeError):run(model,owner,rows,lambda:{},backend,lambda e:None,Path(folder),'reference',b'fixed',counters)
                data=json.loads((Path(folder)/'context-regression.json').read_text())
                self.assertFalse(data['completed']);self.assertEqual(len(data['records']),1)
        with self.assertRaises(RuntimeError):require_backend({**EXPECTED,'threads':1})

if __name__=='__main__':unittest.main()
