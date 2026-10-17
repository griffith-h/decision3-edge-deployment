"""Reuse the identical benchmark protocol with the direct-write packed engine."""
import setup_helpers,runpy,sys,json,hashlib,os
from pathlib import Path
import merged_adapter
from scattered_adapter import PackedEngine
from setup_helpers import configure as configure_old
def configure(engine):
    if not isinstance(engine,PackedEngine):configure_old(engine)
setup_helpers.configure=configure
merged_adapter.PackedEngine=PackedEngine
script=sys.argv.pop(1)
runpy.run_path('/work/'+script,run_name='__main__')
if script=='bench.py':
    mode=sys.argv[sys.argv.index('--mode')+1] if '--mode' in sys.argv else 'baseline'
    cpu=sys.argv[sys.argv.index('--compile')+1] if '--compile' in sys.argv else 'none'
    name=mode+'-'+cpu
elif script=='paired.py':
    mode=sys.argv[sys.argv.index('--mode')+1];cpu=sys.argv[sys.argv.index('--compile')+1] if '--compile' in sys.argv else 'none';name='paired-'+mode+'-'+cpu
else:name='packing-unit'
if os.environ.get('D3_COMPILE_DYNAMIC')=='0' and script in ('bench.py','paired.py') and cpu!='none':name+='-static'
p=Path('/work/logs/'+name+'.json');d=json.loads(p.read_text());d['packed_backend']='native-scatter'
d['native_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/work').iterdir() if p.suffix in ('.cpp','.so')}
p.rename(p.with_name('scatter-'+p.name));p.with_name('scatter-'+p.name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
