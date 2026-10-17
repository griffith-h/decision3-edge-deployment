"""Keep completed signed apt downloads for subsequent disposable containers."""
import shutil,time
from pathlib import Path
dest=Path('/work/apt-cache');dest.mkdir(exist_ok=True)
def apt_running():
    for p in Path('/proc').glob('[0-9]*/comm'):
        try:
            if p.read_text().strip()=='apt-get':return True
        except (FileNotFoundError,PermissionError):pass
    return False
while True:
    for p in Path('/var/cache/apt/archives').glob('*.deb'):
        target=dest/p.name
        if not target.exists():
            try:shutil.copyfile(p,target)
            except FileNotFoundError:pass
    if not apt_running():break
    time.sleep(1)
print('CACHED_COMPILER_DEBS',len(list(dest.glob('*.deb'))),flush=True)
