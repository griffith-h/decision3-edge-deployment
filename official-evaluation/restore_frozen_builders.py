"""Reproduce frozen, non-indexed legacy rows without modifying the upstream kit.

The current SGD builder fixes historical leakage, but the frozen 0.3 files still
contain the old rows; SGD and RouterBench are not in the 0.3 index. All files
must match the official manifest. This never changes gold or model results.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

from decision_index.suite.build.adapters_scored import routerbench
from decision_index.suite.build.layout import Layout

LEGACY = '2d573f852627b7e1e3180d4b6549c90b75e81b29'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', default='output/decision-index-audit/work')
    p.add_argument('--kit', default='output/decision-index-audit/kit')
    a = p.parse_args()
    if sys.version_info < (3, 12):
        raise RuntimeError('RouterBench means require Python >=3.12 float summation to match frozen files')
    layout = Layout(a.work)
    path = Path(a.kit)
    source = subprocess.check_output(['git', '-C', str(path), 'show',
                                     LEGACY + ':decision_index/suite/build/adapters_mechanical.py'])
    legacy = layout.root.parent / 'adapters-mechanical-frozen-v1.py'
    legacy.write_bytes(source)
    runpy.run_path(str(legacy))['sgd'](layout)
    routerbench(layout)
    manifest = json.loads((path / 'hub/manifest.json').read_text())
    checks = []
    for benchmark in manifest['benchmarks']:
        if benchmark['catalog_id'] not in (6, 10):
            continue
        for row in benchmark['sources']:
            file = layout.root / row['path']
            value = hashlib.sha256(file.read_bytes()).hexdigest()
            assert value == row['sha256'], file
            checks.append({'file': file.name, 'sha256': value, 'matches': True})
    report = {'purpose': 'match frozen non-indexed rows; not a new task definition',
              'legacy_builder_commit': LEGACY, 'legacy_builder_sha256': hashlib.sha256(source).hexdigest(),
              'python': sys.version, 'checks': checks}
    (layout.root.parent / 'legacy-builder-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
