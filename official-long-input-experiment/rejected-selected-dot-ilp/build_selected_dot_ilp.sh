#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gcc -Wall -Wextra -O3 -ffp-contract=off -fopenmp -std=c11 -march=armv8-a+simd -shared -fPIC selected_dot_ilp.c -lm -o libselected_dot_ilp.so
python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
files=['selected_dot_ilp.c','build_selected_dot_ilp.sh','libselected_dot_ilp.so']
Path('selected-dot-build.json').write_text(json.dumps({'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
 'flags':'-Wall -Wextra -O3 -ffp-contract=off -fopenmp -std=c11 -march=armv8-a+simd -shared -fPIC -lm',
 'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
 'scope':'Independent outputs interleaved; each output preserves sequential K FMA and final +0/BF16 boundary. No fast-math or reassociation.'},indent=2)+'\n')
PY
