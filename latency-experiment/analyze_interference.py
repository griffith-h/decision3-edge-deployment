"""Correlate HTTP wall-clock intervals with untouched production health probes."""
from collections import defaultdict
import argparse
import datetime
import json
from pathlib import Path

root = Path(__file__).resolve().parent/'logs'
p = argparse.ArgumentParser()
p.add_argument('--http', default='http-native.json')
p.add_argument('--health', default='production-health-history.json')
p.add_argument('--monitor', default='host-http-monitor.json')
p.add_argument('--output', default='interference-analysis.json')
a = p.parse_args()
http = json.loads((root/a.http).read_text())
health = json.loads((root/a.health).read_text())
monitor = json.loads((root/a.monitor).read_text()) if a.monitor != 'none' else []
def timestamp(value):
    # Docker emits nanoseconds; microseconds suffice for this overlap check.
    return datetime.datetime.fromisoformat(value.rstrip('Z')[:26]).replace(tzinfo=datetime.timezone.utc).timestamp()
windows = [dict(start=timestamp(x['Start']), end=timestamp(x['End']), exit_code=x['ExitCode'])
           for x in health['Log']]
slow = []
for i, record in enumerate(http['requests']):
    if record['seconds'] < 1:
        continue
    start, end = record['started_at_unix'], record['ended_at_unix']
    overlap = [w for w in windows if w['start'] < end and start < w['end']]
    ticks = defaultdict(int)
    for item in monitor:
        if start-.1 < item['unix'] < end+.1:
            for process in item['active']:
                ticks[(process['pid'], process['comm'])] += process['ticks']
    slow.append(dict(request=i+1, **record, overlapping_health_probes=overlap,
                     process_cpu_ticks=[dict(pid=k[0], comm=k[1], ticks=v)
                                        for k,v in sorted(ticks.items(), key=lambda x:x[1], reverse=True)[:8]]))
result = dict(slow_requests=slow, slow_count=len(slow),
              slow_overlapping_health=sum(bool(r['overlapping_health_probes']) for r in slow),
              production_health_windows=windows,
              http_source=a.http, health_source=a.health,
              health_probes_during_http=sum(w['start'] < http['requests'][-1]['ended_at_unix']
                  and http['requests'][0]['started_at_unix'] < w['end'] for w in windows),
              caveat='Temporal overlap is correlation. Production probes were not disabled or changed. '
                     + ('Process monitor ran on A53 cores 0-3. ' if monitor else 'No host process monitor was used for this run. ')
                     + 'HTTP uses Mac wall time; health/monitor use board wall time.')
(root/a.output).write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
