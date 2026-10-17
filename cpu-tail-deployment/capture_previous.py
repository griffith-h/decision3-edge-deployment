"""Read-only deployment baseline; store hashes of container configs, never environment secrets."""
import hashlib,json,subprocess,urllib.request,datetime
from pathlib import Path
root=Path(__file__).resolve().parent
code="""import json,subprocess,hashlib
from pathlib import Path
c=json.loads(subprocess.check_output(['docker','inspect','decision3-edge-optimized'],text=True))[0]
h=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True).encode()).hexdigest()
host=c['HostConfig'].copy();host.pop('RestartPolicy',None)
p=Path('/opt/decision3-edge-optimized-20261017-v1')
print(json.dumps({'id':c['Id'],'started_at':c['State']['StartedAt'],'running':c['State']['Running'],'image':c['Image'],'config_sha256':h(c['Config']),'host_config_without_restart_sha256':h(host),'restart':c['HostConfig']['RestartPolicy'],'source_hashes':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and '__pycache__' not in str(f)}}))
"""
r=subprocess.run(['ssh','root@192.168.1.164','python3 -'],input=code,text=True,capture_output=True,check=True)
s=json.loads(r.stdout)
with urllib.request.urlopen('http://192.168.1.164:8900/health') as r:s['health']=json.load(r)
s['at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
(root/'previous-state.json').write_text(json.dumps(s,indent=2)+'\n')
print('PREVIOUS',s['id'],s['health']['deployment']['release'])
