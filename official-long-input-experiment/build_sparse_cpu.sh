#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gcc -Wall -Wextra -O3 -ffp-contract=off -fopenmp -std=c11 -march=armv8-a+simd -shared -fPIC sparse_cpu.c -lm -o libsparse_cpu.so
python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
files=['sparse_cpu.c','build_sparse_cpu.sh','libsparse_cpu.so']
Path('sparse-cpu-build.json').write_text(json.dumps({'compiler':subprocess.check_output(['gcc','--version'],text=True).splitlines()[0],
 'flags':'-Wall -Wextra -O3 -ffp-contract=off -fopenmp -std=c11 -march=armv8-a+simd -shared -fPIC -lm',
 'scope':'Own bounded BF16 gather; optional public CBLAS approximate repair requires full-model gates; no SDK or model changes',
 'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}},indent=2)+'\n')
PY
