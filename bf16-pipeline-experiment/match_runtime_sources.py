"""Match local final runtime/benchmark sources to the frozen board manifest."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent;manifest=json.loads((P/'logs/artifacts-after.json').read_text());assert manifest['unchanged'];prefix='/opt/decision3-edge-bf16-pipeline-20261018/'
names=['cache_owner.py','accepted_owner.py','split_owner.py','split_mlp.py','native_owner.py','native_mlp.py','native_split.cpp','native_mlp.cpp','selected_owner.py','fp32_owner.py','precision_readout_owner.py','readout_owner.py','last_readout.py','precise_mlp.py','core_owner.py','npu_attention.py','execution.json','selection.json','selected.json','config.json','run.sh','formal_native.py','regression_native.py','profile_native.py','short_formal.py','build-selected.sh','build-native.sh','build-native-split.sh']
r={'passed':True,'files':{}}
for name in names:
 sha=hashlib.sha256((P/name).read_bytes()).hexdigest();assert sha==manifest['files'][prefix+name]['sha256'],name;r['files'][name]=sha
(P/'logs/runtime-source-match.json').write_text(json.dumps(r,indent=2)+'\n');print('RUNTIME SOURCE MATCH',len(names))
