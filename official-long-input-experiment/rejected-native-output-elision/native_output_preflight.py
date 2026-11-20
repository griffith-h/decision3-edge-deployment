"""No live Owner: verify owned ABI/build and real SDK output elision guards."""
import hashlib,json,unittest
from pathlib import Path

def run(emit,out):
    emit({'event':'ready','case':'native-output-preflight','model_loaded':False})
    emit({'event':'request_begin','phase':'native_output_tests','budget_seconds':120})
    root=Path(__file__).parent;manifest=json.loads((root/'linear-precision-build.json').read_text())
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:raise ValueError('Native output build identity mismatch')
    import test_native_output_elision_candidate,test_native_output_policy
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (test_native_output_elision_candidate,test_native_output_policy))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    (out/'native-output-preflight.json').write_text(json.dumps({'passed':result.wasSuccessful(),'tests_run':result.testsRun,'build':manifest,'model_loaded':False},indent=2)+'\n')
    if not result.wasSuccessful():raise AssertionError('Checked native output preflight failed')
    emit({'event':'complete','tests':result.testsRun,'model_loaded':False})
