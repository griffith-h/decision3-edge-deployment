"""Read-only process CPU sampling on A53 cores; no process arguments collected."""
import argparse
import json
from pathlib import Path
import time

p = argparse.ArgumentParser()
p.add_argument('--seconds', type=int, default=150)
p.add_argument('--output', required=True)
a = p.parse_args()
previous = {}
records = []
deadline = time.monotonic()+a.seconds
while time.monotonic() < deadline:
    active, current = [], {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = path.read_text()
            end = text.rfind(')')
            comm = text[text.index('(')+1:end]
            fields = text[end+2:].split()
            pid = int(path.parent.name)
            ticks = int(fields[11])+int(fields[12])
            start_ticks = int(fields[19])
            cpu = int(fields[36])
            identity = (pid, start_ticks)
            current[identity] = ticks
            old = previous.get(identity)
            if old is not None and ticks > old:
                active.append(dict(pid=pid, comm=comm, cpu=cpu, ticks=ticks-old))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    records.append(dict(unix=time.time(), active=active))
    previous = current
    time.sleep(.1)
Path(a.output).write_text(json.dumps(records)+'\n')
