from pathlib import Path
import json,hashlib,subprocess,difflib
P=Path(__file__).resolve().parent
long=json.loads((P.parent/'long-sequence-experiment/logs/artifacts.json').read_text())
row=json.loads((P.parent/'row-projection-experiment/logs/artifacts.json').read_text())
code='''from pathlib import Path
import hashlib,json,subprocess,re
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not subprocess.check_output(['docker','ps','--filter','name=decision3-long-tail','--format','{{.Names}}'],text=True).strip()
p=Path('/opt/decision3-edge-long-tail-20261018')
sources={f.name:h(f) for f in p.iterdir() if f.is_file() and (f.suffix in ('.py','.c','.sh','.so') or f.name in ('config.json','selected.json'))}
for prefix,files in DEPENDENCIES.items():
 for name,digest in files.items():assert h(Path(prefix)/name)==digest,(prefix,name)
header=Path('/opt/decision3-edge-matmul/sdk/rknn_api.h');text=header.read_text();start=text.index('RKNN_NPU_CORE_AUTO');snippet=text[start-30:start+500]
for token,value in [('RKNN_NPU_CORE_AUTO',0),('RKNN_NPU_CORE_0',1),('RKNN_NPU_CORE_1',2)]:assert re.search(token+r'\\s*=\\s*'+str(value)+r'\\b',text),token
archive={str(f.relative_to(p/'logs/profile-sources')):h(f) for f in (p/'logs/profile-sources').rglob('*') if f.is_file() and '__pycache__' not in str(f)}
print(json.dumps({'sources':sources,'previous_dependencies_verified':DEPENDENCIES,'profile_source_archive':archive,'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],'sdk_core_enum':{'path':str(header),'sha256':h(header),'excerpt':snippet}}))
'''
deps={'/opt/decision3-edge-long-sequence-20261018':long['sources'],'/opt/decision3-edge-row-projection-20261018':row['sources'],'/opt/decision3-edge-mlp-length-20261018':row['mlp_dependencies']}
code=code.replace('DEPENDENCIES',repr(deps))
r=subprocess.run(['ssh','root@192.168.1.164','python3 -'],input=code,text=True,capture_output=True)
if r.returncode:raise RuntimeError(r.stderr)
a=json.loads(r.stdout)
for name,digest in a['sources'].items():
 if not name.endswith('.so'):assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
for name,digest in a['profile_source_archive'].items():assert hashlib.sha256((P/'logs/profile-sources'/name).read_bytes()).hexdigest()==digest,name
changes=[]
for src,dst in [('long-sequence-experiment/long_attention.c','tiled_attention.c'),('long-sequence-experiment/long_scan.c','tuned_scan.c'),('remaining-500ms-experiment/scan_fast.c','recurrent_scan.c')]:
 old=P.parent/src
 if old.exists():changes+=difflib.unified_diff(old.read_text().splitlines(True),(P/dst).read_text().splitlines(True),fromfile=src,tofile=dst)
(P/'logs/native-source.diff').write_text(''.join(changes));(P/'logs/artifacts.json').write_text(json.dumps(a,indent=2)+'\n');print('ARTIFACTS PASS',len(a['sources']))
