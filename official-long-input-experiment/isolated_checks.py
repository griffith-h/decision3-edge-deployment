"""Run SDK/unit checks outside the live model owner's Python process.

Tests may change Torch backend flags, RNG and transformer registries. The child
runs serially while the parent is idle, with identical interpreter/sys.path and
container CPU/SDK limits; it never inherits or loads the parent's model owner.
"""
import json,subprocess,sys

CHILD='''import importlib,json,sys,unittest
from pathlib import Path
sys.path[:]=json.loads(sys.argv[1])
names=json.loads(sys.argv[2]);target=Path(sys.argv[3])
suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module(name)) for name in names)
result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(suite)
state={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'modules':names}
torch=sys.modules.get('torch')
if torch is not None:state['child_backend_after_tests']={'mkldnn':torch.backends.mkldnn.enabled,'threads':torch.get_num_threads(),'float32_precision':torch.get_float32_matmul_precision()}
target.write_text(json.dumps(state,indent=2)+'\\n')
raise SystemExit(0 if result.wasSuccessful() else 1)
'''

def run_checks(names,result_path,search_path=None):
    if not names:raise ValueError('Boundary check module list cannot be empty')
    command=[sys.executable,'-c',CHILD,json.dumps(list(sys.path if search_path is None else search_path)),json.dumps(names),str(result_path)]
    result=subprocess.run(command,check=False)
    if result.returncode:raise RuntimeError('Isolated candidate boundary checks failed; see child stderr and result JSON')
    return json.loads(result_path.read_text())
