"""Development stop and exception paths must restore the model method."""
import hashlib,json,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
import coverage_scan

class CoverageTests(unittest.TestCase):
    def test_missing_answers_and_nonfinite_values_are_blocking(self):
        self.assertEqual(coverage_scan.response_issues({'answers':{}},{'q':{}}),['answer_fields'])
        self.assertEqual(coverage_scan.response_issues({'answers':{'q':{'score':float('nan')}}},{'q':{}}),['nonfinite_output'])
    def fixture(self,folder):
        root=Path(folder);inputs=root/'inputs';inputs.mkdir();out=root/'out';out.mkdir()
        rows=[{'id':f'coverage-{i:04d}','payload':{'state':'synthetic','questions':{'q':{}}}} for i in range(2)]
        data=json.dumps(rows)+'\n';(inputs/'coverage-inputs.json').write_text(data)
        manifest={'input_file_sha256':hashlib.sha256(data.encode()).hexdigest(),'source_sample_rows_sha256':'fixed',
            'cases':[{'id':r['id'],'run_id':str(i)} for i,r in enumerate(rows)]}
        (inputs/'coverage-manifest.json').write_text(json.dumps(manifest))
        encoder=object();model=types.SimpleNamespace(encoder=encoder,logits=lambda seq,counts:'original')
        owner=types.SimpleNamespace(last_readout=types.SimpleNamespace(encoder=encoder))
        original=model.logits;calls=[]
        def system_one(**kwargs):
            calls.append(kwargs);model.logits([range(5)],[2]);return {'answers':{'q':{'choice':'yes'}}}
        model.system_one=system_one
        return inputs,out,model,owner,original,calls
    def test_slow_completed_request_stops_before_next_input_and_restores_method(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs,out,model,owner,old,calls=self.fixture(folder);events=[]
            with patch.object(coverage_scan,'Path',side_effect=lambda name:inputs/Path(name).name),patch.object(coverage_scan.time,'perf_counter',side_effect=[0,31]):
                coverage_scan.run(model,owner,lambda:{},events.append,out)
            self.assertEqual(len(calls),1);self.assertIs(model.logits,old)
            status=json.loads((out/'coverage-status.json').read_text())
            self.assertFalse(status['completed']);self.assertEqual(status['stop_reason'],'slow')
            self.assertEqual(events[-1]['event'],'complete')
    def test_inference_exception_preserves_active_shapes_and_restores_method(self):
        with tempfile.TemporaryDirectory() as folder:
            inputs,out,model,owner,old,calls=self.fixture(folder)
            def fail(**kwargs):model.logits([range(7)],[2]);raise RuntimeError('injected inference failure')
            model.system_one=fail
            with patch.object(coverage_scan,'Path',side_effect=lambda name:inputs/Path(name).name):
                with self.assertRaisesRegex(RuntimeError,'inference failure'):
                    coverage_scan.run(model,owner,lambda:{},lambda event:None,out)
            self.assertIs(model.logits,old)
            active=json.loads((out/'active-case.json').read_text())
            self.assertEqual(active['batches'][0]['lengths'],[7]);self.assertFalse(active['completed'])

if __name__=='__main__':unittest.main()
