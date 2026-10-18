"""Hash immutable experimental/dependency artifacts only while tests are stopped."""
import argparse,json,subprocess,datetime
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['before','after']);args=parser.parse_args();P=Path(__file__).resolve().parent
code="""import pathlib,hashlib,json,subprocess
assert not subprocess.check_output(['docker','ps','--filter','name=decision3-fusion-pipeline-test','--format','{{.Names}}'],text=True).strip(), 'Stop test before hashing'
roots=['/opt/decision3-edge-fusion-pipeline-20261018','/opt/decision3-edge-long-tail-20261018','/opt/decision3-edge-long-sequence-20261018','/opt/decision3-edge-row-projection-20261018','/opt/decision3-edge-mlp-length-20261018','/opt/decision3-edge-matmul/sdk','/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e']
r={}
for root in roots:
 p=pathlib.Path(root)
 for f in sorted(p.iterdir()):
  if f.is_file() and (root.startswith('/root/.local/state/') or f.suffix in ('.py','.c','.cpp','.inc','.so','.rknn','.sh','.h','.hpp','.safetensors') or f.name in ('execution.json','selected.json','config.json')):
   h=hashlib.sha256()
   with f.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
   r[str(f)]={'bytes':f.stat().st_size,'sha256':h.hexdigest()}
print(json.dumps(r))
"""
r=subprocess.run(['ssh','root@192.168.1.164','python3 -'],input=code,text=True,capture_output=True,check=True);d={'files':json.loads(r.stdout),'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
if args.phase=='after':
 old=json.loads((P/'logs/artifacts-before.json').read_text());assert d['files']==old['files'],'Source/binary changed during final acceptance';d['unchanged']=True
(P/f'logs/artifacts-{args.phase}.json').write_text(json.dumps(d,indent=2)+'\n');print('ARTIFACTS',args.phase,len(d['files']),d.get('unchanged'))
