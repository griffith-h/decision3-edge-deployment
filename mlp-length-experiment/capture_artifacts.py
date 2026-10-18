"""Run only after benchmark containers exit; hash reads must not overlap timing."""
from pathlib import Path
import json,hashlib,subprocess
P=Path(__file__).resolve().parent
local={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(P.glob('*.rknn'))}
assert len(local)==24,'Compile both 12-layer graph sets first'
local['mlp-layers.pt']=hashlib.sha256((P.parent/'cpu-tail-experiment/mlp-layers.pt').read_bytes()).hexdigest()
code="""from pathlib import Path
import hashlib,json,subprocess
assert not subprocess.check_output(['docker','ps','--filter','name=decision3-mlp-length','--format','{{.Names}}'],text=True).strip(),'Benchmarks still running'
p=Path('/opt/decision3-edge-mlp-length-20261018')
files={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.iterdir()) if f.is_file() and f.suffix in ('.py','.sh','.cpp','.rknn','.so','.pt')}
h=Path('/opt/decision3-edge-matmul/sdk/rknn_api.h');files['sdk/rknn_api.h']=hashlib.sha256(h.read_bytes()).hexdigest();print(json.dumps(files))
"""
r=subprocess.run(['ssh','root@192.168.1.164','python3 -'],input=code,text=True,capture_output=True,check=True);board=json.loads(r.stdout)
for name,digest in local.items():assert board[name]==digest,name
(P/'logs/local-artifacts.json').write_text(json.dumps(local,indent=2)+'\n');(P/'logs/board-artifacts.json').write_text(json.dumps(board,indent=2)+'\n');print('ARTIFACT PASS',len(local),'local/board hashes match')
