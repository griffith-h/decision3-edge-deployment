"""Real child-process isolation, including failing checks, without Torch/model."""
import os,sys,tempfile,unittest
from pathlib import Path
from isolated_checks import run_checks

class IsolationTests(unittest.TestCase):
    def test_child_mutation_does_not_escape_and_failures_remain_visible(self):
        name='D3_TEST_ISOLATION_MARKER';old=os.environ.get(name)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'child_good.py').write_text('import os,unittest\nos.environ["'+name+'"]="child"\nclass Tests(unittest.TestCase):\n def test_environment(self):self.assertEqual(os.environ["'+name+'"],"child")\n')
            (root/'child_bad.py').write_text('import os,unittest\nos.environ["'+name+'"]="failed-child"\nclass Tests(unittest.TestCase):\n def test_failure(self):self.fail("expected child failure")\n')
            paths=[folder,*sys.path]
            result=run_checks(['child_good'],root/'passed.json',paths)
            self.assertTrue(result['passed']);self.assertEqual(result['tests_run'],1);self.assertEqual(os.environ.get(name),old)
            with self.assertRaisesRegex(RuntimeError,'Isolated candidate boundary checks failed'):
                run_checks(['child_bad'],root/'failed.json',paths)
            self.assertEqual(os.environ.get(name),old)
            import json
            failed=json.loads((root/'failed.json').read_text());self.assertFalse(failed['passed']);self.assertEqual(failed['failures'],1)
if __name__=='__main__':unittest.main()
