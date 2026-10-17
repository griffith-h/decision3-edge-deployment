"""Run on RK host: freeze all runtime inputs before staging; never modifies old services."""
import argparse, hashlib, json, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--release',default='/opt/decision3-edge-graph-20261018-v1');a=p.parse_args()
out=Path(a.release)
if out.exists():raise SystemExit('Release exists; use a fresh version directory')
source=Path('/opt/decision3-edge-cpu-tail-20261017')
expected=json.loads((source/'logs/formal-graph-single.json').read_text())['sources']
verified=json.loads((source/'experiment-result.json').read_text())['verified_runtime_hashes']
for name,digest in verified.items():
 actual=hashlib.sha256((source/name).read_bytes()).hexdigest()
 assert actual==digest==expected[name],name
out.mkdir();(out/'work').mkdir();(out/'service').mkdir();(out/'deps').mkdir()
for f in source.iterdir():
 if f.is_file() and f.suffix in ('.py','.c','.cpp','.h','.so','.rknn'):
  shutil.copy2(f,out/'work'/f.name)
# Existing imports address these mounted aliases. Freeze their code/libraries too.
old=Path('/opt/decision3-edge-optimized-20261017-v1')
shutil.copytree(old,out/'deps/release')
aliases={'references':'decision3-edge-matmul','combined':'decision3-edge-bridge-parallel-20261017',
 'scan':'decision3-edge-scan-20261017','remainder':'decision3-edge-remainder-20261017',
 'projection':'decision3-edge-projection-compile-20261017','previous':'decision3-edge-remaining-500ms-20261017',
 'attn':'decision3-edge-attention-chunk-20261017'}
for alias,name in aliases.items():
 root=Path('/opt')/name
 for f in root.rglob('*'):
  if any(x in ('__pycache__','compiler-env','toolkit','.venv') for x in f.relative_to(root).parts):continue
  if f.is_file() and f.suffix in ('.py','.c','.cpp','.h','.so','.json'):
   dest=out/'deps'/alias/f.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
for name in ('server.py','runtime.py','config.json','request.json'):
 shutil.copy2(Path(__file__).parent/name,out/'service'/name)
for name in ('run.sh','rollback.sh','cutover.py'):
 shutil.copy2(Path(__file__).parent/name,out/name)
shutil.copy2(old/'image-id.txt',out/'image-id.txt')
manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'SHA256SUMS').write_text(''.join(f'{v}  {k}\n' for k,v in sorted(manifest.items())))
print('PACKAGED',out,len(manifest),'files')
