"""Finalize an already acquired/normalized corpus using unchanged upstream builders."""
import argparse
import json
from pathlib import Path

from decision_index.suite.build import rebuild
from decision_index.suite.build.layout import Layout, NAMES
from decision_index.suite.download import from_local


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--work', default='output/decision-index-audit/work')
    p.add_argument('--suite', default='output/decision-index-audit/suite-0.3')
    a = p.parse_args()
    layout = Layout(a.work)
    missing = [name for names in NAMES.values() for name in names
               if not (layout.normalized / (name + '.jsonl')).is_file()]
    if missing:
        raise RuntimeError('Normalization still incomplete: ' + ', '.join(missing))
    result = rebuild.main(a.work, skip_download=True, skip_normalize=True, edition='0.3')
    (layout.root.parent / 'corpus-rebuild.json').write_text(json.dumps(result, indent=2) + '\n')
    assert result['release_v1']['byte_identical'], 'Original public corpus differs from upstream hash'
    assert result['rows_byte_identical'] and result['added_byte_identical'] and result['gsm8k']['byte_identical']
    verification = from_local(
        a.suite, layout.suite / 'release-v2-rebuilt/selected-rows.jsonl.gz',
        added=layout.suite / 'release-v2-rebuilt/added-rows.jsonl.gz',
        gsm8k=layout.suite / 'release-v3-rebuilt/gsm8k-rows.jsonl.gz', edition='0.3')
    assert verification['exclusions_match']
    (layout.root.parent / 'corpus-verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    print(json.dumps(verification), flush=True)


if __name__ == '__main__':
    main()
