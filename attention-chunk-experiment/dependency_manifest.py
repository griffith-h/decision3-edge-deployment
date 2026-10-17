"""Run on the board only after all experiments have exited."""
import hashlib,json
from pathlib import Path
roots=['/opt/decision3-edge-optimized-20261017-v1','/opt/decision3-edge-bridge-parallel-20261017','/opt/decision3-edge-scan-20261017','/opt/decision3-edge-remainder-20261017','/opt/decision3-edge-projection-compile-20261017','/opt/decision3-edge-remaining-500ms-20261017']
paths=set()
for folder in roots:
    for p in Path(folder).rglob('*'):
        if p.is_file() and p.suffix in ('.py','.c','.cpp','.so','.sh','.h') and not any(k in p.parts for k in ('__pycache__','inductor-cache','apt-cache','logs')):paths.add(p)
sdk=Path('/opt/decision3-edge-matmul/sdk')
paths.update(p for p in sdk.iterdir() if p.is_file() and p.suffix in ('.h','.so'))
p=Path('/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e')
paths.update(p/name for name in ('d3_runtime.py','config.json'))
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
Path(__file__).with_name('dependency-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('dependency files',len(manifest))
