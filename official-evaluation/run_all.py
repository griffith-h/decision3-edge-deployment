"""Run requested stages on the frozen sample, then restore the retained service."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import board_engine
from decision_index.runner import run
from decision_index.suite.io import sha256_file
from score_sample import score
from stages import PIN, STAGES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sample', required=True)
    p.add_argument('--out', default='output/decision-index-audit/runs')
    p.add_argument('--host', default='root@192.168.1.164')
    p.add_argument('--remote', default='/opt/decision3-official-evaluation-final-20261019')
    p.add_argument('--stop-service', action='store_true')
    p.add_argument('--stages', nargs='+', choices=[s[0] for s in STAGES])
    a = p.parse_args()
    selected = [s[0] for s in STAGES if not a.stages or s[0] in a.stages]
    sample = Path(a.sample).resolve()
    manifest = json.loads(sample.with_suffix('.manifest.json').read_text())
    assert sha256_file(sample) == manifest['sample_file_sha256']
    assert sha256_file(sample, gunzip=sample.suffix == '.gz') == manifest['sample_rows_sha256']
    root = Path(a.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol_files = ('board.py', 'board_engine.py', 'host_container.py', 'stages.py',
                      'score_sample.py', 'run_all.py', 'requirements.lock',
                      'evaluation_verdict.py', 'serve_final.py')
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                     for name in protocol_files}
    signature = {'sample_rows_sha256': manifest['sample_rows_sha256'],
                 'model_revision': PIN, 'kit_commit': manifest['kit_commit'], 'harness': source_hashes,
                 'host_python': sys.version, 'requested_stages': selected, 'remote': a.remote}
    frozen = root / 'protocol.json'
    if frozen.exists():
        assert json.loads(frozen.read_text()) == signature, 'Protocol changed; use a new run directory'
    else:
        frozen.write_text(json.dumps(signature, indent=2) + '\n')
    state_path = root / 'run-status.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    def save():
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(state_path)

    def close_workers():
        failures = []
        for engine in board_engine.LIVE:
            try:
                engine.close()
            except Exception as exc:
                failures.append(type(exc).__name__)
        board_engine.LIVE.clear()
        if failures:
            state['_worker_cleanup_errors'] = failures
            save()

    def ssh(command, **kw):
        return subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                               a.host, command], check=True, **kw)

    def interrupted(signum, frame):
        raise KeyboardInterrupt('Interrupted; restoring retained service')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    service = root / 'retained-service.json'
    restore = False
    try:
        if a.stop_service:
            if service.exists():
                old = json.loads(service.read_text())
            else:
                inspection = ssh('docker inspect decision3-edge-graph', capture_output=True, text=True)
                info = json.loads(inspection.stdout)[0]
                # Keep identity, not environment values or credentials.
                old = {'id': info['Id'], 'image': info['Image'], 'was_running': info['State']['Running'],
                       'started_at': info['State']['StartedAt'], 'name': info['Name']}
                service.write_text(json.dumps(old, indent=2) + '\n')
            current = json.loads(ssh('docker inspect decision3-edge-graph', capture_output=True, text=True).stdout)[0]
            assert current['Id'] == old['id'], 'Retained service identity changed'
            restore = old['was_running']
            ssh('docker stop -t 90 decision3-edge-graph', stdout=subprocess.DEVNULL)
        for stage, *_ in STAGES:
            if a.stages and stage not in a.stages:
                continue
            folder = root / stage
            folder.mkdir(exist_ok=True)
            if state.get(stage, {}).get('sample_complete'):
                continue
            state[stage] = {'status': 'running', 'started_at': time.time()}
            save()
            try:
                with (folder / 'runner.log').open('a') as log:
                    result = run('board_engine:BoardEngine', {'stage': stage, 'out': str(folder),
                                 'host': a.host, 'remote': a.remote},
                                 sample, folder, corpus_sha256=manifest['sample_rows_sha256'],
                                 compact=False, log=lambda message: print(message, file=log, flush=True))
                report = score(sample, folder / 'results.jsonl', folder, stage)
                state[stage].update(status='complete' if report['sample_complete'] else 'incomplete',
                                    sample_complete=report['sample_complete'], counts=report['counts'],
                                    accuracy=report['labelled_field_accuracy']['accuracy'],
                                    board_method=report['board_method'])
            except KeyboardInterrupt:
                state[stage].update(status='interrupted', sample_complete=False)
                raise
            except Exception as exc:
                (folder / 'failure.txt').write_text(traceback.format_exc())
                state[stage].update(status='failed', sample_complete=False, error_type=type(exc).__name__)
                if (folder / 'results.jsonl').exists():
                    try:
                        score(sample, folder / 'results.jsonl', folder, stage)
                    except Exception:
                        (folder / 'scoring-failure.txt').write_text(traceback.format_exc())
            finally:
                close_workers()
                state[stage]['finished_at'] = time.time()
                save()
                print(json.dumps({'stage': stage, **state[stage]}), flush=True)
    finally:
        close_workers()
        if restore:
            ssh('docker start decision3-edge-graph', stdout=subprocess.DEVNULL)
            ready = False
            for _ in range(90):
                checked = ssh("docker inspect --format '{{.State.Health.Status}}' decision3-edge-graph",
                              capture_output=True, text=True)
                if checked.stdout.strip() == 'healthy':
                    ready = True
                    break
                time.sleep(2)
            state['_service'] = {'action': 'restarted retained graph-half container after evaluation',
                                 'healthy': ready, 'at': time.time()}
            save()
            if not ready:
                raise RuntimeError('Retained service restarted but did not become healthy')


if __name__ == '__main__':
    main()
