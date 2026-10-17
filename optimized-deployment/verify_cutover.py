"""One-time deployment verification; SSH operations are explicit and recorded."""
import datetime,json,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
HOST='192.168.1.164'
BASE='http://'+HOST+':8900'
RELEASE='/opt/decision3-edge-optimized-20261017-v1'
body=(ROOT.parent/'request.json').read_bytes()
def ssh(command):
    return subprocess.check_output(['ssh',HOST,command],text=True)
def call(path,payload=None):
    with urllib.request.urlopen(urllib.request.Request(BASE+path,data=payload,headers={'Content-Type':'application/json'}),timeout=60) as r:return json.load(r)
def ready(optimized):
    for _ in range(180):
        try:
            h=call('/health')
            assert h['status']=='ready'
            assert ('deployment' in h)==optimized
            if optimized:
                assert h['deployment']['release']=='2026-10-17-three-thread-v1'
                assert h['deployment']['torch_threads']==3
            return h
        except (OSError,AssertionError):time.sleep(2)
    raise RuntimeError('Service readiness failed')
# This script is a one-time controlled transition, not an idempotent installer.
assert json.loads((ROOT/'staging-lifecycle.json').read_text())['passed']
original=json.loads((ROOT/'original-response.json').read_text())['response']
print(ssh('docker rm -v decision3-edge-optimized-staging && docker stop -t 60 decision3-edge-matmul && bash '+RELEASE+'/run.sh decision3-edge-optimized 8900 unless-stopped'),flush=True)
try:
    first=ready(True)
    first_response=call('/v1/systemone',body)
    assert first_response['answers']['kind']['choice']=='code'
except Exception:
    ssh('docker stop -t 60 decision3-edge-optimized; docker start decision3-edge-matmul')
    raise
print('Optimized 8900 ready; testing rollback',flush=True)
rollback_log=ssh('bash '+RELEASE+'/rollback.sh')
rollback_health=ready(False)
rollback_response=call('/v1/systemone',body)
assert rollback_response==original,'Original output changed on rollback'
print('Rollback original response exactly equal; reactivating optimized',flush=True)
print(ssh('docker stop -t 60 decision3-edge-matmul && docker start decision3-edge-optimized'),flush=True)
final_health=ready(True)
containers=json.loads(ssh('docker inspect decision3-edge-optimized decision3-edge-matmul decision3-edge'))
container_summary=[{'name':c['Name'].lstrip('/'),'id':c['Id'],'image_id':c['Image'],'running':c['State']['Running'],'started_at':c['State']['StartedAt'],'exit_code':c['State']['ExitCode'],'oom_killed':c['State']['OOMKilled'],'restart_policy':c['HostConfig']['RestartPolicy']['Name'],'cpuset':c['HostConfig']['CpusetCpus'],'memory_bytes':c['HostConfig']['Memory'],'init':c['HostConfig']['Init'],'healthcheck':c['Config'].get('Healthcheck'),'ports':c['HostConfig']['PortBindings']} for c in containers]
old=json.loads((ROOT/'original-container.json').read_text())[0]
assert containers[1]['Id']==old['Id'] and not containers[1]['State']['Running']
assert containers[1]['Config']==old['Config'] and containers[1]['HostConfig']==old['HostConfig']
assert containers[0]['State']['Running'] and not containers[2]['State']['Running']
result={'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,
 'first_optimized_health':first,'first_optimized_response':first_response,'rollback_health':rollback_health,
 'rollback_response_exactly_preserved':True,'rollback_log':rollback_log,'final_health':final_health,
 'containers':container_summary,'original_container_id_and_config_preserved':True,
 'scope':'Staging API + in-flight request SIGTERM test; deploy 8900; actual rollback and original response equality; restart same optimized container.'}
(ROOT/'cutover-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('CUTOVER_AND_ROLLBACK_PASSED',flush=True)
