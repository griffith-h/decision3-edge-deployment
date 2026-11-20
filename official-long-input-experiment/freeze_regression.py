"""Freeze existing self-built regression payloads; comparison outputs remain local."""
import hashlib,json
from pathlib import Path
root=Path('output/decision-index-audit/iterations/r001')
native_path=Path('bf16-pipeline-experiment/logs/regression-native.json')
current_path=Path('fusion-pipeline-experiment/logs/references.json')
native=json.loads(native_path.read_text())['cases'];current=json.loads(current_path.read_text())['cases']
assert len(native)==34 and set(native)==set(current)
inputs=[];refs={}
for name,record in native.items():
 payload={k:v for k,v in record['request'].items() if k in ('state','questions','images','videos')}
 inputs.append({'id':name,'payload':payload});refs[name]={'native':record['output'],'current':current[name]['output']}
for name,data in [('regression-inputs.json',inputs),('regression-references.json',refs)]:
 p=root/name
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(data,ensure_ascii=False)+'\n')
(root/'regression-manifest.json').write_text(json.dumps({'cases':len(inputs),'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (native_path,current_path)},'input_file_sha256':hashlib.sha256((root/'regression-inputs.json').read_bytes()).hexdigest()},indent=2)+'\n')
print('FROZEN_REGRESSION',len(inputs))
