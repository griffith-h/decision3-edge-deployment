"""Read-only provenance capture after all timed processes exit."""
from pathlib import Path
import subprocess,json,hashlib
P=Path(__file__).resolve().parent
code="""import subprocess,hashlib,json
from pathlib import Path
assert not subprocess.check_output(['docker','ps','--filter','name=decision3-row-projection','--format','{{.Names}}'],text=True).strip()
p=Path('/opt/decision3-edge-row-projection-20261018')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={f.name:h(f) for f in p.iterdir() if f.is_file() and f.suffix in ('.py','.sh')}
q=Path('/opt/decision3-edge-mlp-length-20261018')
mlp={f.name:h(f) for f in q.iterdir() if f.is_file() and f.suffix in ('.py','.so','.rknn')}
print(json.dumps({'sources':files,'mlp_dependencies':mlp}))
"""
r=subprocess.run(['ssh','root@192.168.1.164','python3 -'],input=code,text=True,capture_output=True,check=True);out=json.loads(r.stdout)
for name,digest in out['sources'].items():assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
(P/'logs/artifacts.json').write_text(json.dumps(out,indent=2)+'\n');print('SOURCE HASH PASS',len(out['sources']))
