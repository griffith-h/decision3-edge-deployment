"""Read-only verification that profiling left the deployed service intact."""
import datetime,hashlib,json,statistics,subprocess,time,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
before=json.loads((root/'service-before.json').read_text())
remote='''import hashlib,json,subprocess
from pathlib import Path
c=json.loads(subprocess.check_output(['docker','inspect','decision3-edge-optimized'],text=True))[0]
remaining=subprocess.check_output(['docker','ps','--filter','name=decision3-bottleneck','--format','{{.Names}}'],text=True).strip()
release=Path('/opt/decision3-edge-optimized-20261017-v1')
hashes={name:hashlib.sha256((release/name).read_bytes()).hexdigest() for name in ['server.py','cpu_attention_optimized.py','native_norm.py','native_norm.c']}
print(json.dumps({'container':c,'remaining':remaining,'hashes':hashes}))
'''
reply=subprocess.run(['ssh','192.168.1.164','python3 -'],input=remote,text=True,capture_output=True,check=True)
state=json.loads(reply.stdout);c=state['container']
assert c['Id']==before['container_id'] and c['State']['StartedAt']==before['started_at']
assert c['Config']==before['config'] and c['HostConfig']==before['host_config']
assert c['State']['Running'] and c['State']['Health']['Status']=='healthy' and not state['remaining']
for name,sha in state['hashes'].items():
    local=root.parent/('optimized-deployment' if name=='server.py' else 'latency-experiment')/name
    assert hashlib.sha256(local.read_bytes()).hexdigest()==sha
base='http://192.168.1.164:8900';body=(root.parent/'request.json').read_bytes();records=[]
for _ in range(5):
    t=time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(base+'/v1/systemone',data=body,headers={'Content-Type':'application/json'}),timeout=30) as r:response=json.load(r)
    records.append(time.perf_counter()-t)
    assert response==before['response']
with urllib.request.urlopen(base+'/health',timeout=10) as r:health=json.load(r)
assert health['deployment']['torch_threads']==3
result={'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'container_id':c['Id'],'started_at':c['State']['StartedAt'],
 'id_start_time_config_and_host_config_unchanged':True,'source_sha256':state['hashes'],'responses_exactly_equal':True,'health':health,
 'inference_http_seconds':records,'median':statistics.median(records),'min':min(records),'max':max(records),'remaining_test_containers':[],
 'scope':'Final verification after all isolated profiling/capability probes completed; existing service never stopped or reconfigured.'}
(root/'service-preservation-final.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('health','source_sha256')},ensure_ascii=False))
