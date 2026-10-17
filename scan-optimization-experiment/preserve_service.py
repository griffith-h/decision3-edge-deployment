"""Read-only before/after capture, run on the developer Mac."""
import argparse,datetime,json,statistics,subprocess,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('phase',choices=['before','after']);a=p.parse_args()
root=Path(__file__).resolve().parent
remote='''import hashlib,json,subprocess
from pathlib import Path
c=json.loads(subprocess.check_output(['docker','inspect','decision3-edge-optimized'],text=True))[0]
release=Path('/opt/decision3-edge-optimized-20261017-v1')
hashes={str(p.relative_to(release)):hashlib.sha256(p.read_bytes()).hexdigest() for p in release.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
tests=subprocess.check_output(['docker','ps','--filter','name=decision3-scan','--format','{{.Names}}'],text=True).splitlines()
print(json.dumps({'container_id':c['Id'],'started_at':c['State']['StartedAt'],'config':c['Config'],'host_config':c['HostConfig'],'running':c['State']['Running'],'health_status':c['State']['Health']['Status'],'hashes':hashes,'remaining_test_containers':tests}))
'''
state=json.loads(subprocess.run(['ssh','192.168.1.164','python3 -'],input=remote,text=True,capture_output=True,check=True).stdout)
base='http://192.168.1.164:8900';records=[];responses=[]
body=(root.parent/'request.json').read_bytes()
for _ in range(3 if a.phase=='before' else 5):
    t=time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(base+'/v1/systemone',data=body,headers={'Content-Type':'application/json'}),timeout=30) as r:responses.append(json.load(r))
    records.append(time.perf_counter()-t)
with urllib.request.urlopen(base+'/health',timeout=10) as r:state['health']=json.load(r)
state.update(verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),http_seconds=records,median=statistics.median(records),responses=responses)
assert state['running'] and state['health_status']=='healthy' and not state['remaining_test_containers']
if a.phase=='after':
    old=json.loads((root/'service-before.json').read_text())
    for k in ('container_id','started_at','config','host_config','hashes'):assert state[k]==old[k],k
    assert all(r==old['responses'][0] for r in responses)
    state['passed']=True
(root/('service-'+a.phase+'.json')).write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
print(a.phase,state['container_id'],state['median'],state.get('passed'))
