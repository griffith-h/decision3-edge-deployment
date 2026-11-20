import hashlib,json,tempfile,unittest
from pathlib import Path
from context_regression import EXPECTED
from check_context_regression import freeze,check,read_run

class ContextGateTests(unittest.TestCase):
    def response(self,value=.5,tokens=1000):
        return {'model':'fixed','usage':{'input_tokens':tokens},'answers':{'q':{'score':value}}}
    def fixture(self,root,variant):
        p=root/variant;p.mkdir()
        inputs=root/'regression-inputs.json';inputs.write_text(json.dumps([{'id':str(i),'payload':{'questions':{'q':{}}}} for i in range(34)]))
        input_digest=hashlib.sha256(inputs.read_bytes()).hexdigest()
        data={'completed':True,'variant':variant,'context':EXPECTED,'case_ids':[str(i) for i in range(34)],'input_sha256':input_digest,
            'records':[{'id':str(i),'response':self.response(tokens=615 if i==1 else 1000),'seconds':1,'backend_before':EXPECTED,'backend_after':EXPECTED,'npu_calls_delta':([0,11,0,0] if variant=='baseline' and i==1 else [0]*4)} for i in range(34)]}
        files={'execution.json':{'exit_code':0,'request_completed':True,'retained_healthy':True,'cleanup_errors':[],'worker_final_state':{'OOMKilled':False}},
            'context-regression.json':data,'identity.json':{},'lab-sources.json':{},'runtime-backend-before-cleanup.json':EXPECTED,'lifecycle.json':{'handles':0},
            ('isolated-boundary-checks.json' if variant.startswith('full-batch') else 'isolated-request-scope-checks.json'):{'passed':True}}
        for n,v in files.items():(p/n).write_text(json.dumps(v))
        return p
    def test_independent_references_keep_history_and_reject_original_and_protected_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cpu=self.fixture(root,'reference');native=self.fixture(root,'baseline');candidate=self.fixture(root,'full-batch-sparse-request-scope')
            np=native/'context-regression.json';nd=json.loads(np.read_text());nd['records'][0]['response']=self.response(.502);np.write_text(json.dumps(nd))
            target=root/'references.json';refs=freeze(cpu,native,target)
            with self.assertRaises(AssertionError):freeze(cpu,native,target)
            history=root/'history.json';history.write_text(json.dumps(refs['references']));old=history.read_bytes()
            p=candidate/'context-regression.json';data=json.loads(p.read_text());data['records'][0]['response']=self.response(.511);data['records'][1]['response']=self.response(.500002,615);p.write_text(json.dumps(data))
            result=check(candidate,target,history)
            self.assertFalse(result['passed']);self.assertEqual(result['passed_cases'],32);self.assertEqual(history.read_bytes(),old)
            self.assertIn('original numerical gate',result['rows'][0]['failure']);self.assertIn('protected native identity',result['rows'][1]['failure'])
    def test_historical_original_limit_is_not_erased_by_new_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cpu=self.fixture(root,'reference');native=self.fixture(root,'baseline');candidate=self.fixture(root,'full-batch-sparse-request-scope')
            target=root/'references.json';refs=freeze(cpu,native,target)
            history=root/'history.json';old=refs['references'];old['0']['current']=self.response(.489);history.write_text(json.dumps(old))
            result=check(candidate,target,history)
            self.assertFalse(result['passed']);self.assertEqual(result['passed_cases'],33)
            self.assertIsNone(result['rows'][0]['failure']);self.assertFalse(result['rows'][0]['historical_core_passed'])
    def test_context_and_identity_mismatch_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cpu=self.fixture(root,'reference');native=self.fixture(root,'baseline')
            p=cpu/'context-regression.json';data=json.loads(p.read_text());data['records'][0]['backend_after']={**EXPECTED,'mkldnn':False};p.write_text(json.dumps(data))
            with self.assertRaises(RuntimeError):read_run(cpu,'reference')
            data['records'][0]['backend_after']=EXPECTED;p.write_text(json.dumps(data));(native/'identity.json').write_text('{"different":true}')
            with self.assertRaises(AssertionError):freeze(cpu,native,root/'references.json')

if __name__=='__main__':unittest.main()
