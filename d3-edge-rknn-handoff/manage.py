#!/usr/bin/env python3
"""Portable RK3576 installer/operator. Python standard library on the host."""
import argparse
import datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
OWNER = 'io.d3-edge.handoff'


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def load_config(path):
    c = json.loads(path.read_text())
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', c['container_name']):
        raise ValueError('Invalid container_name')
    ipaddress.IPv4Address(c['bind_host'])
    if not isinstance(c['port'], int) or not 1 <= c['port'] <= 65535:
        raise ValueError('Invalid port')
    if not re.fullmatch(r'\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*', c['cpu_set']):
        raise ValueError('Invalid cpu_set')
    if c['restart'] not in ('no', 'unless-stopped'):
        raise ValueError('restart must be no or unless-stopped')
    data = Path(c['data_dir']).expanduser()
    c['data_dir'] = str((ROOT / data).resolve()) if not data.is_absolute() else str(data.resolve())
    return c


def inspect(c):
    p = subprocess.run(['docker', 'container', 'inspect', c['container_name']], capture_output=True, text=True)
    if p.returncode:
        run(['docker', 'info'], stdout=subprocess.DEVNULL)
        return None
    value = json.loads(p.stdout)[0]
    if value['Config'].get('Labels', {}).get(OWNER) != '1':
        raise RuntimeError('Container name is occupied by an unrelated container; choose another container_name')
    return value


def config_hash(c):
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def doctor(c):
    if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
        raise RuntimeError('Run on RK3576 Linux ARM64, not an x86 PC or Mac')
    compatible = Path('/proc/device-tree/compatible').read_bytes().replace(b'\0', b' ').decode()
    if 'rk3576' not in compatible.lower():
        raise RuntimeError('This package is validated for RK3576 only')
    if not Path('/dev/dri').is_dir():
        raise RuntimeError('Missing /dev/dri; use a board OS with the RKNPU driver enabled')
    if (os.cpu_count() or 0) < 8:
        raise RuntimeError('Expected 8 CPUs for this RK3576 configuration')
    cpus = set()
    for part in c['cpu_set'].split(','):
        ends = list(map(int, part.split('-')))
        first, last = ends[0], ends[-1]
        if first > last:
            raise ValueError('Invalid CPU range')
        cpus.update(range(first, last + 1))
    if not cpus or max(cpus) >= (os.cpu_count() or 0):
        raise ValueError('cpu_set exceeds available CPUs')
    run(['docker', 'info'], stdout=subprocess.DEVNULL)
    driver = Path('/sys/kernel/debug/rknpu/version')
    print('PASS: Linux ARM64 / RK3576 / Docker / /dev/dri')
    print(driver.read_text().strip() if driver.exists() else 'Driver version unavailable here; baseline was 0.9.8')
    print('Data directory:', c['data_dir'])


def base_run(c):
    return ['docker', 'run', '--rm', '--init', '--cpuset-cpus', c['cpu_set'], '--memory', c['memory'],
            '--memory-swap', c['memory'], '--device', '/dev/dri']


def request(c, route, body=None):
    host = '127.0.0.1' if c['bind_host'] == '0.0.0.0' else c['bind_host']
    headers = {'Content-Type': 'application/json'}
    if os.environ.get('DECISION_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['DECISION_API_KEY']
    req = urllib.request.Request(f'http://{host}:{c["port"]}{route}', headers=headers,
                                 data=None if body is None else json.dumps(body, ensure_ascii=False).encode())
    with urllib.request.urlopen(req, timeout=120 if body else 5) as response:
        return json.load(response)


def wait_ready(c):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        state = inspect(c)
        if not state or not state['State']['Running']:
            raise RuntimeError('Service exited during startup; run logs')
        try:
            health = request(c, '/health')
            if health['status'] == 'ready':
                print(json.dumps(health, ensure_ascii=False))
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(2)
    raise RuntimeError('Startup not ready within 240s; container retained, run logs')


def numeric(x, path=''):
    result = {}
    if isinstance(x, dict):
        for key, child in x.items():
            if key not in ('confidence', 'legend'):
                result.update(numeric(child, path + '/' + key))
    elif isinstance(x, (int, float)):
        result[path] = x
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.json')
    parser.add_argument('action', choices=['init', 'doctor', 'build', 'download', 'start', 'stop', 'remove', 'status', 'logs', 'smoke', 'check', 'benchmark'])
    args = parser.parse_args()
    if args.action == 'init':
        with args.config.open('x') as out:
            out.write((ROOT / 'config.example.json').read_text())
        print('Created', args.config)
        return
    c = load_config(args.config)
    data = Path(c['data_dir'])
    action = args.action
    if action == 'doctor':
        doctor(c)
    elif action == 'build':
        doctor(c)
        run(['docker', 'build', '--platform', 'linux/arm64', '-t', c['image'], str(ROOT)])
    elif action == 'download':
        (data / 'models').mkdir(parents=True, exist_ok=True)
        cmd = ['docker', 'run', '--rm', '-e', 'HF_HUB_OFFLINE=0', '-e', 'TRANSFORMERS_OFFLINE=0',
               '-v', f'{data / "models"}:/models', '-v', f'{ROOT / "tools"}:/setup:ro']
        for key in ('HF_TOKEN', 'HF_ENDPOINT'):
            if key in os.environ:
                cmd.extend(['-e', key])
        run(cmd + [c['image'], '/setup/download_model.py'])
    elif action == 'start':
        doctor(c)
        if not (data / 'models/d3-edge/MODEL_MANIFEST.json').is_file():
            raise RuntimeError('Pinned model missing; run download first')
        existing = inspect(c)
        if existing:
            if existing['Config']['Labels'].get('io.d3-edge.config') != config_hash(c):
                raise RuntimeError('Configuration changed: stop, remove, then start this managed container')
            if not existing['State']['Running']:
                run(['docker', 'start', c['container_name']])
        else:
            cmd = ['docker', 'run', '-d', '--init', '--name', c['container_name'], '--label', OWNER + '=1',
                   '--label', 'io.d3-edge.config=' + config_hash(c), '--restart', c['restart'],
                   '--cpuset-cpus', c['cpu_set'], '--memory', c['memory'], '--memory-swap', c['memory'],
                   '--device', '/dev/dri', '-p', f'{c["bind_host"]}:{c["port"]}:8900',
                   '-v', f'{data / "models"}:/models:ro', '--log-opt', 'max-size=10m', '--log-opt', 'max-file=3',
                   '--health-cmd', 'python -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8900/health\",timeout=5).read()"',
                   '--health-interval', '30s', '--health-timeout', '10s', '--health-start-period', '180s']
            if 'DECISION_API_KEY' in os.environ:
                cmd.extend(['-e', 'DECISION_API_KEY'])
            run(cmd + [c['image']])
        wait_ready(c)
    elif action in ('stop', 'remove', 'status', 'logs'):
        existing = inspect(c)
        if not existing:
            print('No managed container created yet')
            return
        if action == 'stop':
            run(['docker', 'stop', c['container_name']])
        elif action == 'remove':
            if existing['State']['Running']:
                raise RuntimeError('Stop the container before remove; model files will remain')
            run(['docker', 'rm', c['container_name']])
        elif action == 'logs':
            run(['docker', 'logs', '--tail', '80', c['container_name']])
        else:
            print(json.dumps({'status': existing['State']['Status'], 'health': existing['State'].get('Health', {}).get('Status'),
                              'restart': existing['HostConfig']['RestartPolicy']['Name'], 'port': c['port']}, ensure_ascii=False))
    elif action == 'smoke':
        assert request(c, '/health')['status'] == 'ready'
        body = json.loads((ROOT / 'app/request.json').read_text())
        start = time.perf_counter()
        response = request(c, '/v1/systemone', body)
        elapsed = time.perf_counter() - start
        reference = json.loads((ROOT / 'evidence/reference-benchmark.json').read_text())['cases']['single']['cpu']['response']
        actual, expected = numeric(response['answers']), numeric(reference['answers'])
        assert actual.keys() == expected.keys()
        delta = max(abs(actual[k] - expected[k]) for k in actual)
        assert response['answers']['kind']['choice'] == reference['answers']['kind']['choice'] and delta <= 0.01
        record = {'passed': True, 'seconds': elapsed, 'max_delta_vs_reference_optimized_cpu': delta, 'response': response}
        output = data / 'checks'
        output.mkdir(parents=True, exist_ok=True)
        (output / 'smoke.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(record, ensure_ascii=False, indent=2))
    elif action in ('check', 'benchmark'):
        doctor(c)
        existing = inspect(c)
        restore = bool(existing and existing['State']['Running'])
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
        output = data / 'checks' / stamp
        output.mkdir(parents=True)
        try:
            if restore:
                run(['docker', 'stop', c['container_name']])
            cmd = base_run(c) + ['-v', f'{output}:/work/logs', '-v', f'{data / "models"}:/models:ro', c['image']]
            if action == 'check':
                run(cmd + ['-c', 'import subprocess,sys; [subprocess.run([sys.executable,p],check=True) for p in ("/work/micro_test.py","/work/boundary_test.py")]'])
            else:
                run(cmd + ['/work/benchmark.py', '--model', '/models/d3-edge', '--selection', 'all', '--repeats', '3'])
            print('PASS; new results:', output)
        finally:
            if restore:
                run(['docker', 'start', c['container_name']])
                wait_ready(c)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, AssertionError, subprocess.CalledProcessError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(1)
