"""Read-only before/after capture of the CURRENT graph-half production service."""
import argparse,datetime,hashlib,json,subprocess,time,urllib.request,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('phase',choices=['before','after']);a=p.parse_args();P=Path(__file__).resolve().parent
code="""import hashlib,json,subprocess
from pathlib import Path
c=json.loads(subprocess.check_output(['docker','inspect','decision3-edge-graph'],text=True))[0]
h=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
p=Path('/opt/decision3-edge-graph-20261018-v1')
files={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and '__pycache__' not in str(f)}
print(json.dumps({'id':c['Id'],'started_at':c['State']['StartedAt'],'config_hash':h(c['Config']),'hostconfig_hash':h(c['HostConfig']),'files':files,'running':c['State']['Running'],'health':c['State']['Health']['Status'],'tests':subprocess.check_output(['docker','ps','--filter','name=decision3-row-projection','--format','{{.Names}}'],text=True).splitlines()}))
"""
r=subprocess.run(['ssh','root@192.168.1.164','python3 -'],input=code,text=True,capture_output=True,check=True);d=json.loads(r.stdout);body=(P.parent/'request.json').read_bytes();d['responses']=[];d['seconds']=[]
for _ in range(3 if a.phase=='before' else 5):
 t=time.perf_counter()
 with urllib.request.urlopen(urllib.request.Request('http://192.168.1.164:8900/v1/systemone',data=body,headers={'Content-Type':'application/json'}),timeout=30) as r:d['responses'].append(json.load(r))
 d['seconds'].append(time.perf_counter()-t)
assert d['running'] and d['health']=='healthy' and not d['tests']
if a.phase=='after':
 old=json.loads((P/'service-before.json').read_text())
 for k in ('id','started_at','config_hash','hostconfig_hash','files'):assert d[k]==old[k],k
 assert all(r==old['responses'][0] for r in d['responses']);d['passed']=True
d['at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(P/('service-'+a.phase+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');print(a.phase,d['id'][:12],statistics.median(d['seconds']),d.get('passed'))
