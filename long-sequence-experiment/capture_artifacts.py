from pathlib import Path
import subprocess,json,hashlib,difflib
P=Path(__file__).resolve().parent
code="""import subprocess,json,hashlib
from pathlib import Path
assert not subprocess.check_output(['docker','ps','--filter','name=decision3-long-sequence','--format','{{.Names}}'],text=True).strip()
p=Path('/opt/decision3-edge-long-sequence-20261018')
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={f.name:h(f) for f in p.iterdir() if f.is_file() and (f.suffix in ('.py','.c','.sh','.so') or f.name in ('config.json','natural-request.txt'))}
row=Path('/opt/decision3-edge-row-projection-20261018');old=PREVIOUS_EVIDENCE
for name,digest in old['sources'].items():assert h(row/name)==digest,name
mlp=Path('/opt/decision3-edge-mlp-length-20261018')
for name,digest in old['mlp_dependencies'].items():assert h(mlp/name)==digest,name
print(json.dumps({'sources':sources,'previous_row_source_hashes':old['sources'],'previous_mlp_hashes':old['mlp_dependencies'],'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0]}))
"""
previous=json.loads((P.parent/'row-projection-experiment/logs/artifacts.json').read_text())
code=code.replace('PREVIOUS_EVIDENCE',repr(previous))
r=subprocess.run(['ssh','root@192.168.1.164','python3 -'],input=code,text=True,capture_output=True)
if r.returncode:raise RuntimeError(r.stderr)
r=json.loads(r.stdout)
for name,digest in r['sources'].items():
    if not name.endswith('.so'):assert hashlib.sha256((P/name).read_bytes()).hexdigest()==digest,name
before=json.loads((P/'service-before.json').read_text())
for name,key in [('cpu-tail-experiment/cpu_attention.c','work/cpu_attention.c'),('attention-chunk-experiment/chunk_blas2.c','deps/attn/chunk_blas2.c')]:assert hashlib.sha256((P.parent/name).read_bytes()).hexdigest()==before['files'][key]
diffs=[]
for src,dst in [('cpu-tail-experiment/cpu_attention.c','long_attention.c'),('attention-chunk-experiment/chunk_blas2.c','long_scan.c')]:diffs+=list(difflib.unified_diff((P.parent/src).read_text().splitlines(True),(P/dst).read_text().splitlines(True),fromfile=src,tofile=dst))
(P/'logs/native-source.diff').write_text(''.join(diffs));r['native_sources_match_frozen_origins']=True
measured=json.loads((P/'logs/measured-artifacts.json').read_text());changed=[]
for name,digest in measured['sources'].items():
    if r['sources'][name]!=digest:changed.append(name)
assert set(changed)=={'long_kernels.py','test_kernels.py'},changed
for name in changed:assert hashlib.sha256((P/'logs/pre-scale-guard'/name).read_bytes()).hexdigest()==measured['sources'][name]
guarddiff=list(difflib.unified_diff((P/'logs/pre-scale-guard/long_kernels.py').read_text().splitlines(True),(P/'long_kernels.py').read_text().splitlines(True),fromfile='measured/long_kernels.py',tofile='final/long_kernels.py'))
(P/'logs/scale-guard.diff').write_text(''.join(guarddiff));r['post_measurement_guard_changes']=changed;r['native_binaries_unchanged_since_measurement']=True
(P/'logs/artifacts.json').write_text(json.dumps(r,indent=2)+'\n');print('ARTIFACT HASH PASS',len(r['sources']))
