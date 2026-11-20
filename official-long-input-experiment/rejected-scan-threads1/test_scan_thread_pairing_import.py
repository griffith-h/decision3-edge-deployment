import importlib.util,sys,types,unittest
from pathlib import Path

class ExplicitCompareImportTests(unittest.TestCase):
    def test_historical_module_name_collision_cannot_shadow_private_compare(self):
        previous=sys.modules.get('summarize');sys.modules['summarize']=types.ModuleType('wrong_historical_module')
        try:
            path=Path(__file__).with_name('scan_thread_pairing.py');spec=importlib.util.spec_from_file_location('isolated_pair_import_test',path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            def response(v):return {'model':'fixed','usage':{'input_tokens':1},'answers':{'q':{'score':v}}}
            result=module.paired_compare(response(.505),response(.5),False)
            self.assertAlmostEqual(result['max_delta'],.005);self.assertTrue(result['choices_unchanged'])
        finally:
            if previous is None:sys.modules.pop('summarize',None)
            else:sys.modules['summarize']=previous

if __name__=='__main__':unittest.main()
