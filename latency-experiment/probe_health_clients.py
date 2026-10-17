"""Measure client process cost without changing the running health check."""
import json
from pathlib import Path
import subprocess
import time

python_code = '''import time, resource
t = time.perf_counter()
import urllib.request, json
urllib.request.urlopen('http://127.0.0.1:8900/health', timeout=5).read()
r = resource.getrusage(resource.RUSAGE_SELF)
print(json.dumps(dict(inside_seconds=time.perf_counter()-t, process_user_seconds=r.ru_utime, process_system_seconds=r.ru_stime)))
'''
clients = [('python', ['python', '-c', python_code]),
           ('curl', ['/usr/bin/curl', '-fsS', '--max-time', '5', '-o', '/dev/null',
                     '-w', '{"request_seconds":%{time_total}}\n', 'http://127.0.0.1:8900/health'])]
records = []
for _ in range(3):
    for name, command in clients:
        start = time.perf_counter()
        run = subprocess.run(['docker', 'exec', 'decision3-edge-matmul', *command],
                             text=True, capture_output=True, timeout=20, check=True)
        records.append(dict(client=name, whole_docker_exec_seconds=time.perf_counter()-start,
                            **json.loads(run.stdout)))
Path('/opt/decision3-edge-latency/logs/health-client-cost.json').write_text(json.dumps(records, indent=2)+'\n')
print(json.dumps(records), flush=True)
