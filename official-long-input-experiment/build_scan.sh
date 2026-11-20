#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd \
  long_scan_extended.c -lm -o liblong_scan_extended.so
python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
files=['long_scan_extended.c','liblong_scan_extended.so','build_scan.sh']
Path('scan-build.json').write_text(json.dumps({
 'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
 'flags':'-Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd -lm',
 'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
 'scope':'full state-carry block scan, bounded coefficients'
},indent=2)+'\n')
PY
