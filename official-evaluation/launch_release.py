"""Run on RK host after evaluation: stage, activate with rollback, or restore old."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

OLD = 'decision3-edge-graph'
NEW = 'decision3-edge-native-bf16'
STAGING = NEW + '-staging'
RELEASE = '2026-10-19-native-bf16-v1'
LABEL = 'decision3.evaluated-release'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def docker(*args, **kwargs):
    # Docker inspect can contain credentials. Callers must never print it.
    return subprocess.check_output(['docker', *args], text=True, **kwargs).strip()


def inspect(name):
    result = subprocess.run(['docker', 'inspect', name], text=True, capture_output=True)
    if result.returncode:
        return None
    return json.loads(result.stdout)[0]


def package(root):
    root = root.resolve()
    if (root / 'INCOMPLETE').exists():
        raise RuntimeError('Incomplete release')
    manifest = json.loads((root / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise RuntimeError('Package hash mismatch: ' + relative)
    return manifest, digest(root / 'manifest.json')


def api_environment(old):
    env = os.environ.copy()
    values = [s.split('=', 1)[1] for s in old['Config'].get('Env', [])
              if s.startswith('DECISION_API_KEY=')]
    env['DECISION_API_KEY'] = values[-1] if values else ''
    return env


def health(port, env):
    headers = {'Authorization': 'Bearer ' + env['DECISION_API_KEY']} if env.get('DECISION_API_KEY') else {}
    request = urllib.request.Request(f'http://192.168.1.164:{port}/health', headers=headers)
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def ready(port, env):
    for _ in range(240):
        try:
            result = health(port, env)
            if result['status'] == 'ready' and result['deployment']['release'] == RELEASE:
                return result
        except (OSError, KeyError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError('Final service did not become healthy')


def launch(root, manifest, marker, env, name, port):
    if inspect(name):
        raise RuntimeError('Container already exists; inspect before retry: ' + name)
    cmd = ['run', '-d', '--init', '--name', name, '--label', LABEL + '=' + marker,
           '--restart', 'no' if port == 8901 else 'unless-stopped',
           '--cpuset-cpus', '4-7', '--memory', '8g', '--memory-swap', '8g',
           '--ulimit', 'core=0', '--ulimit', 'nofile=4096:4096', '--device', '/dev/dri',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'TOKENIZERS_PARALLELISM=false',
           '-e', 'OMP_NUM_THREADS=3', '-e', 'OMP_THREAD_LIMIT=4', '-e', 'OPENBLAS_NUM_THREADS=3',
           '-e', 'DECISION_API_KEY', '-p', f'192.168.1.164:{port}:8900',
           '--log-opt', 'max-size=10m', '--log-opt', 'max-file=3',
           '--health-cmd', 'curl --fail --silent --max-time 5 --output /dev/null http://127.0.0.1:8900/health',
           '--health-interval', '30s', '--health-timeout', '10s', '--health-start-period', '240s']
    for alias, source in manifest['bindings'].items():
        if not Path(source).is_dir():
            raise FileNotFoundError(source)
        cmd += ['--mount', f'type=bind,source={source},target=/{alias},readonly']
    cmd += ['--workdir', '/experiment', '--entrypoint', 'python', manifest['image'],
            '-u', '/evaluation/serve_final.py', '--port', '8900']
    return docker(*cmd, env=env)


def remove_owned(name, marker):
    current = inspect(name)
    if current:
        if current['Config'].get('Labels', {}).get(LABEL) != marker:
            raise RuntimeError('Refusing to remove another container: ' + name)
        docker('stop', '-t', '90', name)
        docker('rm', name)


def restore(old):
    current = inspect(OLD)
    if current is None or current['Id'] != old['Id']:
        raise RuntimeError('Retained container identity changed; cannot restore')
    policy = old['HostConfig']['RestartPolicy']
    restart = policy['Name']
    if restart == 'on-failure' and policy.get('MaximumRetryCount'):
        restart += ':' + str(policy['MaximumRetryCount'])
    docker('update', '--restart=' + restart, OLD)
    if old['State']['Running']:
        docker('start', OLD)
        env = api_environment(old)
        for _ in range(180):
            try:
                result = health(8900, env)
                if result['status'] == 'ready' and result['deployment']['release'] == '2026-10-18-graph-half-v1':
                    return
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(2)
        raise RuntimeError('Retained service restarted but did not become healthy')


def activate(root, proof_path, acceptance_path):
    manifest, marker = package(root)
    proof = json.loads(Path(proof_path).read_text())
    verdict = root / 'service/evaluation-result.json'
    acceptance = json.loads(Path(acceptance_path).read_text())
    if not (proof.get('passed') and proof.get('manifest_sha256') == marker
            and proof.get('port') == 8901 and proof.get('regression_cases', 0) >= 34):
        raise RuntimeError('Full staging HTTP proof required')
    if not (acceptance.get('accepted') and acceptance.get('reason')
            and acceptance.get('verdict_sha256') == digest(verdict)
            and acceptance.get('sample_rows_sha256') == manifest['sample_rows_sha256']):
        raise RuntimeError('Review actual accuracy, failures and speed before activation')
    old = inspect(OLD)
    if not old or not old['State']['Running'] or inspect(NEW):
        raise RuntimeError('Expected retained service running and final name unused')
    env = api_environment(old)
    if health(8901, env)['deployment']['sample_rows_sha256'] != manifest['sample_rows_sha256']:
        raise RuntimeError('Wrong staging endpoint')
    record = {'previous_id': old['Id'], 'previous_restart': old['HostConfig']['RestartPolicy'],
              'manifest_sha256': marker, 'passed': False}
    try:
        remove_owned(STAGING, marker)
        docker('update', '--restart=no', OLD)
        docker('stop', '-t', '90', OLD)
        record['new_id'] = launch(root, manifest, marker, env, NEW, 8900)
        record['health'] = ready(8900, env)
        record['passed'] = True
    except BaseException:
        # Cleanup failure must never prevent restoration of the retained endpoint.
        try:
            remove_owned(NEW, marker)
        finally:
            restore(old)
        raise
    finally:
        (root / 'activation.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['stage', 'activate', 'rollback'])
    p.add_argument('--root', required=True)
    p.add_argument('--proof')
    p.add_argument('--acceptance')
    a = p.parse_args()
    root = Path(a.root).resolve()
    manifest, marker = package(root)
    if a.action == 'stage':
        old = inspect(OLD)
        if not old or not old['State']['Running']:
            raise RuntimeError('Retained service must be running before staging')
        env = api_environment(old)
        launch(root, manifest, marker, env, STAGING, 8901)
        print(json.dumps({'health': ready(8901, env), 'manifest_sha256': marker}))
    elif a.action == 'activate':
        if not a.proof or not a.acceptance:
            p.error('activate requires --proof and --acceptance')
        print(json.dumps(activate(root, a.proof, a.acceptance)))
    else:
        activation = json.loads((root / 'activation.json').read_text())
        old = inspect(OLD)
        if not old or old['Id'] != activation['previous_id']:
            raise RuntimeError('Retained service identity changed')
        remove_owned(NEW, marker)
        old['HostConfig']['RestartPolicy'] = activation['previous_restart']
        old['State']['Running'] = True
        restore(old)
        print(json.dumps({'restored': OLD, 'id': old['Id']}))


if __name__ == '__main__':
    main()
