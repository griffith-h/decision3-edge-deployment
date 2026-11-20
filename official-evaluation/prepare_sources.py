"""Acquire pinned Decision Index sources locally; keep restricted data out of Git.

Run using the isolated environment with the unmodified upstream kit installed.
This does not skip failed sources or label an incomplete corpus official-complete.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import urllib.error

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
from decision_index.suite.build import acquire, adapters_added, rebuild
from decision_index.suite.build.layout import Layout


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', default='output/decision-index-audit/work')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--include-hle', action='store_true')
    p.add_argument('--only', type=int, nargs='+', help='Retry only these source catalog ids')
    a = p.parse_args()
    layout = Layout(a.work)
    logs = layout.root.parent / 'acquisition-logs'
    logs.mkdir(parents=True, exist_ok=True)
    groups = [('base', [n]) for n in sorted(rebuild.BUILDERS)
              if n not in (27, 45)] + [('base', [26, 27])]
    groups = [(k, ns) for k, ns in groups if ns != [26]]
    groups += [('added', [n]) for n in adapters_added.ORDER]
    if a.include_hle:
        groups.append(('base', [45]))
    status = logs / 'status.json'
    if a.only:
        groups = [(k, ns) for k, ns in groups if set(ns) & set(a.only)]
    records = json.loads(status.read_text()) if status.exists() else {}

    def task(kind, numbers):
        name = kind + '-' + '-'.join(map(str, numbers))
        record = {'kind': kind, 'catalog_ids': numbers, 'started_at': time.time()}
        with (logs / (name + '.log')).open('a') as f:
            def log(message):
                print(message, file=f, flush=True)
            try:
                fn = acquire.acquire if kind == 'base' else adapters_added.acquire
                try:
                    fn(layout, numbers, log=log)
                except subprocess.CalledProcessError:
                    if kind != 'base' or 37 not in numbers or shutil.which('git-lfs'):
                        raise
                    # A failed first checkout left only downloaded files in this cache repo.
                    # Disable unavailable LFS filters for this checkout only; the unchanged
                    # upstream acquisition then downloads/verifies its pinned media files.
                    subprocess.run(['git', '-C', str(layout.repos / 'esci'),
                                    '-c', 'filter.lfs.required=false', '-c', 'filter.lfs.smudge=',
                                    '-c', 'filter.lfs.process=', 'checkout', '-f', '-q', 'FETCH_HEAD'], check=True)
                    fn(layout, numbers, log=log)
                    record['transport_fix'] = 'cached checkout without unavailable LFS filter; pinned media verified upstream'
                except (urllib.error.URLError, TimeoutError):
                    if kind != 'base' or numbers != [23]:
                        raise
                    archive = 'https://web.archive.org/web/20161207205250id_/http://www.dgp.toronto.edu/~donovan/cfcolor/cfcolor.zip'
                    pinned = acquire.HTTP['raw/downloads/cfcolor.zip'][1]
                    acquire.http_fetch(layout.downloads / 'cfcolor.zip', archive, pinned, log=log)
                    fn(layout, numbers, log=log)
                    record['transport_fix'] = archive
                record['acquired'] = True
            except Exception as exc:
                traceback.print_exc(file=f)
                record.update(acquired=False, error_type=type(exc).__name__)
        record['finished_at'] = time.time()
        return name, record

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(task, k, ns) for k, ns in groups]
        for future in concurrent.futures.as_completed(futures):
            name, record = future.result()
            records[name] = record
            tmp = status.with_suffix('.tmp')
            tmp.write_text(json.dumps(records, indent=2) + '\n')
            tmp.replace(status)
            print(json.dumps({'source': name, **record}), flush=True)
    failures = [k for k, v in records.items() if not v['acquired']]
    print(json.dumps({'attempted': len(records), 'failed': failures,
                      'hle_pending': not any(45 in r['catalog_ids'] and r['acquired'] for r in records.values())}), flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
