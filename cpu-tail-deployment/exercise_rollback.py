"""Actual rollback and return-to-new rehearsal; only run after successful cutover."""
import datetime,json,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;release='/opt/decision3-edge-graph-20261018-v1'
def ssh(cmd):return subprocess.check_output(['ssh','-n','root@192.168.1.164',cmd],text=True)
def call(path,body=None):
 req=urllib.request.Request('http://192.168.1.164:8900'+path,data=None if body is None else json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=60) as r:return json.load(r)
def wait(expected):
 for _ in range(150):
  try:
   h=call('/health')
   if h['deployment']['release']==expected:return h
  except (OSError,KeyError):pass
  time.sleep(2)
 raise RuntimeError('Readiness timeout '+expected)
proof={'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};body=json.loads((P/'request.json').read_text())
assert call('/health')['deployment']['release']=='2026-10-18-graph-half-v1'
ssh('bash '+release+'/rollback.sh')
proof['old_health']=wait('2026-10-17-three-thread-v1');old=call('/v1/systemone',body)
expected=json.loads((P.parent/'cpu-tail-experiment/service-before.json').read_text())['responses'][0]
assert old==expected;proof['old_response_exact']=True;proof['old_response']=old
proof['new_while_rolled_back']=json.loads(ssh('docker inspect --format "{{json .State}}" decision3-edge-graph'))
assert not proof['new_while_rolled_back']['Running']
try:
 ssh('docker update --restart=no decision3-edge-optimized && docker stop -t 90 decision3-edge-optimized && docker update --restart=unless-stopped decision3-edge-graph && docker start decision3-edge-graph')
 proof['new_health']=wait('2026-10-18-graph-half-v1');new=call('/v1/systemone',body)
 frozen=json.loads((P.parent/'cpu-tail-experiment/logs/formal-graph-single.json').read_text())['cases']['single']['response'];assert new==frozen;proof['new_response_exact']=True
except BaseException:
 ssh('bash '+release+'/rollback.sh');raise
proof['passed']=True
(P/'rollback-rehearsal.json').write_text(json.dumps(proof,indent=2)+'\n');print('ACTUAL ROLLBACK AND RETURN PASS',flush=True)
