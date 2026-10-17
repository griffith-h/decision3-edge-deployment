"""Collect final runtime evidence after the production HTTP verification."""
import datetime,hashlib,json,subprocess,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RELEASE='/opt/decision3-edge-optimized-20261017-v1'
def ssh(command):return subprocess.check_output(['ssh','192.168.1.164',command],text=True)
cutover=json.loads((ROOT/'cutover-verification.json').read_text())
http=json.loads((ROOT.parent/'latency-experiment/logs/deployed-production-http.json').read_text())
assert cutover['passed'] and http['passed'] and http['count']==100
with urllib.request.urlopen('http://192.168.1.164:8900/health',timeout=10) as r:health=json.load(r)
assert health['deployment']['torch_threads']==3 and health['deployment']['OMP_THREAD_LIMIT']=='3'
assert health['deployment']['native_stats']['conv_calls']>=600
raw=json.loads(ssh('docker inspect decision3-edge-optimized decision3-edge-matmul decision3-edge'))
assert raw[0]['State']['Running'] and raw[0]['State']['Health']['Status']=='healthy'
assert not raw[1]['State']['Running'] and not raw[2]['State']['Running']
assert raw[0]['HostConfig']['RestartPolicy']['Name']=='unless-stopped'
assert raw[1]['HostConfig']['RestartPolicy']['Name']=='no'
assert raw[0]['HostConfig']['CpusetCpus']=='4-7'
assert raw[0]['HostConfig']['Init']
old=json.loads((ROOT/'original-container.json').read_text())[0]
assert raw[1]['Id']==old['Id'] and raw[1]['Config']==old['Config'] and raw[1]['HostConfig']==old['HostConfig']
original_hashes=ssh('sha256sum /opt/decision3-edge-matmul/npu_server.py /opt/decision3-edge-matmul/cpu_optimized_server.py /opt/decision3-edge-matmul/rknn_linear.py /opt/decision3-edge-matmul/libmatmul_bridge.so /opt/decision3-edge-matmul/sdk/librknnrt.so')
remote_hashes=ssh('cd '+RELEASE+' && sha256sum server.py run.sh rollback.sh cpu_attention_optimized.py native_norm.py native_norm.c libd3_native_norm.so base/*.py base/*.so base/sdk/*.so')
hashes={line.split()[1]:line.split()[0] for line in remote_hashes.splitlines()}
for name in ('server.py','run.sh','rollback.sh'):
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==hashes[name]
for name in ('cpu_attention_optimized.py','native_norm.py','native_norm.c'):
    assert hashlib.sha256((ROOT.parent/'latency-experiment'/name).read_bytes()).hexdigest()==hashes[name]
for line in original_hashes.splitlines():
    sha,path=line.split(); name=Path(path).name
    if name=='npu_server.py':assert hashlib.sha256((ROOT.parent/'rknn-matmul-experiment'/name).read_bytes()).hexdigest()==sha
    else:assert hashes['base/'+('sdk/' if name=='librknnrt.so' else '')+name]==sha
# No temporary listener is retained. curl returning a connect failure is expected.
try:
    urllib.request.urlopen('http://192.168.1.164:8901/health',timeout=3)
except OSError:closed=True
else:closed=False
assert closed
summaries=[{'name':c['Name'].lstrip('/'),'id':c['Id'],'image_id':c['Image'],'running':c['State']['Running'],'health':c['State'].get('Health',{}).get('Status'),'started_at':c['State']['StartedAt'],'restart_policy':c['HostConfig']['RestartPolicy']['Name'],'exit_code':c['State']['ExitCode'],'oom_killed':c['State']['OOMKilled']} for c in raw]
result={'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,
 'release':'2026-10-17-three-thread-v1','host':'192.168.1.164','port':8900,'release_directory':RELEASE,
 'image_id':raw[0]['Image'],'health':health,'containers':summaries,
 'source_sha256':hashes,'original_source_sha256_output':original_hashes,
 'original_container_id_config_and_source_preserved':True,'actual_rollback_and_reactivation_passed':True,
 'staging_8901_closed':closed,'healthcheck':raw[0]['Config']['Healthcheck'],
 'environment':raw[0]['Config']['Env'],'mounts':raw[0]['Mounts'],
 'http_summary':{k:v for k,v in http.items() if k not in ('requests','health')},
 'verification_commands':['python3 -m py_compile optimized-deployment/*.py latency-experiment/verify_http.py','bash -n optimized-deployment/run.sh optimized-deployment/rollback.sh','python3 optimized-deployment/test_rollback.py','cc -Wall -Wextra -O3 -fPIC -shared -fopenmp -ffp-contract=off native_norm.c -o libd3_native_norm.so -lm','python3 latency-experiment/verify_http.py --count 3 --output deployment-staging-http.json','python3 optimized-deployment/verify_cutover.py','python3 latency-experiment/verify_http.py --base http://192.168.1.164:8900 --count 100 --output deployed-production-http.json'],
 'limits':['115 tokens, single question, concurrency 1, loaded and warmed model only','No new full task accuracy evaluation; selected 7-case computational regression reused with identical kernel hashes','Fresh-machine guide compatibility steps statically reviewed, not fully rerun on a blank machine','Finite latency samples do not establish a hard 1-second bound; 200ms not achieved']}
(ROOT/'deployment-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
(ROOT/'deployed-SHA256SUMS').write_text(remote_hashes)
print(json.dumps(result['http_summary'],ensure_ascii=False))
