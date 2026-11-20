"""Decision Index engine using one persistent RK worker, without HTTP variation."""
import json
import os
from pathlib import Path
import selectors
import shlex
import subprocess
import time

from decision_index.engines.base import Engine, Unsupported
from decision_index.engines.http import CAPACITY_MARKERS

FRAME = b'@@D3EVAL@@'
LIVE = []


class BoardEngine(Engine):
    name = 'rk3576-historical'
    latency = ('Official runner timing includes persistent SSH RPC and JSON handling; '
               'raw_output.board_method_ms separately measures the board system_one method. '
               'Excludes model load and initial two warmups; retains first-shape costs.')

    def __init__(self, stage, out, host='root@192.168.1.164',
                 remote='/opt/decision3-official-evaluation-20261019', timeout=600, **options):
        super().__init__(**options)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.stderr = (self.out / 'board-stderr.log').open('ab')
        self.diag = (self.out / 'board-stdout.log').open('ab')
        self.buffer = b''
        self.timeout = float(timeout)
        self.seq = 0
        self.broken = False
        self.restarts = 0
        self.settings = dict(stage=stage, out=out, host=host, remote=remote, timeout=timeout, **options)
        self.host = host
        self.closed = False
        command = shlex.join(['python3', remote + '/host_container.py', '--stage', stage])
        self.process = subprocess.Popen(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8',
                                         host, command], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, bufsize=0)
        LIVE.append(self)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            ready = self.receive(600)
            assert ready['event'] == 'ready'
            self.provenance = ready['identity']
            self.provenance['kind'] = 'historical-rk3576-main-thread-rpc'
            (self.out / 'board-identity.json').write_text(json.dumps(self.provenance, indent=2) + '\n')
        except BaseException:
            self.close()
            raise

    def receive(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            while b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                # Native libraries can print to stdout independently of Python redirect_stdout.
                start = line.find(FRAME)
                if start >= 0:
                    if start:
                        self.diag.write(line[:start] + b'\n')
                    return json.loads(line[start + len(FRAME):])
                self.diag.write(line + b'\n')
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(max(0, remaining)):
                self.broken = True
                self.close()
                raise TimeoutError('Board request exceeded the fixed timeout; worker retired')
            chunk = os.read(self.process.stdout.fileno(), 65536)
            if not chunk:
                self.broken = True
                code = self.process.poll()
                raise RuntimeError(f'Board worker exited before response, code={code}; inspect board logs')
            self.buffer += chunk

    def __call__(self, state, questions):
        if getattr(self, 'broken', False):
            settings, restarts = self.settings, self.restarts + 1
            identity = self.provenance
            self.close()
            self.__init__(**settings)
            assert self.provenance == identity, 'Historical worker identity changed during restart'
            self.restarts = restarts
        if self.closed:
            raise RuntimeError('Board worker is closed')
        self.seq += 1
        body = {'seq': self.seq, 'payload': {'state': state, 'questions': questions}}
        self.process.stdin.write((json.dumps(body, ensure_ascii=False) + '\n').encode())
        self.process.stdin.flush()
        response = self.receive(self.timeout)
        assert response['seq'] == self.seq, 'RPC sequence mismatch'
        response['worker_restarts'] = getattr(self, 'restarts', 0)
        if response['status'] != 'ok':
            message = response.get('error', '')
            # Same explicit capacity markers as the unmodified official HTTP engine.
            if response.get('exception') in ('ValueError', 'Unsupported') and any(x in message for x in CAPACITY_MARKERS):
                raise Unsupported(message)
            raise RuntimeError(json.dumps(response, ensure_ascii=False))
        return response['response'], response

    def runtime(self):
        return {'board_identity': self.provenance}

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.stdin:
            self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Kill only the dedicated test container; the retained deployment is separate.
            subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', self.host,
                            'docker rm -f decision3-official-eval'], stdout=subprocess.DEVNULL,
                           stderr=self.stderr, timeout=25, check=False)
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.selector.close()
        self.stderr.close()
        self.diag.close()
