#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gcc -Wall -Wextra -O3 -ffp-contract=off -fopenmp -std=c11 -march=armv8-a+simd -shared -fPIC selected_layout_tail.c -lm -o libselected_layout_tail.so
python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
files=['selected_layout_tail.c','build_selected_layout_tail.sh','libselected_layout_tail.so']
Path('selected-layout-tail-build.json').write_text(json.dumps({'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],'flags':'-O3 -ffp-contract=off -fopenmp -std=c11 -march=armv8-a+simd -shared -fPIC -lm','sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}},indent=2)+'\n')
PY
