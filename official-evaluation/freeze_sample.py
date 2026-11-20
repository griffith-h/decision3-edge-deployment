"""Verify the complete upstream corpus, then freeze one complete-group sample."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess

from decision_index.suite.io import Suite, read_jsonl, sha256_file
from decision_index.suite.sample import stratified_sample

KIT_PIN = 'f5a6e4fd85fed05e77878dfc4966d0568bfb45ec'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', default='output/decision-index-audit/suite-0.3')
    p.add_argument('--kit', default='output/decision-index-audit/kit')
    p.add_argument('--out', default='output/decision-index-audit/sample-1000.jsonl.gz')
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20260919)
    a = p.parse_args()
    out = Path(a.out)
    if out.exists() or out.with_suffix('.manifest.json').exists():
        raise FileExistsError('Refusing to replace a frozen sample; choose a new output')
    pin = subprocess.check_output(['git', '-C', a.kit, 'rev-parse', 'HEAD'], text=True).strip()
    assert pin == KIT_PIN
    assert not subprocess.check_output(['git', '-C', a.kit, 'diff', '--name-only'], text=True).strip()
    suite = Suite(a.suite, '0.3')
    verified = suite.verify()
    # Upstream verify reports exclusions separately. Require it too.
    assert verified.get('exclusions_match') is True
    universe = collections.defaultdict(set)
    for row in suite.rows(apply_exclusions=True):
        e = row['_evaluation']
        universe[(e['catalog_id'], e['group_id'])].add(e['run_id'])
    result = stratified_sample(suite, a.n, out, seed=a.seed, complete_groups=True)
    rows = list(read_jsonl(out))
    groups = collections.defaultdict(set)
    for row in rows:
        e = row['_evaluation']
        groups[(e['catalog_id'], e['group_id'])].add(e['run_id'])
    assert all(ids == universe[key] for key, ids in groups.items())
    ids = [r['_evaluation']['run_id'] for r in rows]
    assert len(ids) == len(set(ids))
    manifest = {'scope': 'complete-group stratified sample of verified Decision Index 0.3 public text suite',
                'kit_commit': pin, 'seed': a.seed, 'target_requests': a.n,
                'actual_requests': len(rows), 'complete_groups': True,
                'sample_file_sha256': sha256_file(out), 'sample_rows_sha256': sha256_file(out, gunzip=True),
                'upstream_verification': verified, 'group_count': len(groups),
                'benchmark_counts': dict(collections.Counter(r['_evaluation']['catalog_id'] for r in rows)),
                'run_ids_in_order': ids,
                'payload_sha256_in_order': [r['_evaluation']['payload_sha256'] for r in rows]}
    out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({**result, 'rows_sha256': manifest['sample_rows_sha256'],
                      'groups': len(groups), 'benchmarks': len(manifest['benchmark_counts'])}), flush=True)


if __name__ == '__main__':
    main()
