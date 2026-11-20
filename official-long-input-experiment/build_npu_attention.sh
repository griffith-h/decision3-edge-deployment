#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
g++ -Wall -Wextra -O3 -fopenmp -std=c++17 -march=armv8-a+simd -shared -fPIC \
  -I /opt/decision3-edge-matmul/sdk long_npu_attention.cpp \
  -L /opt/decision3-edge-graph-20261018-v1/deps/release/base/sdk \
  -lrknnrt -Wl,-rpath,/base/sdk -o liblong_npu_attention.so
python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
files=['long_npu_attention.cpp','liblong_npu_attention.so','build_npu_attention.sh']
Path('npu-attention-build.json').write_text(json.dumps({
 'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
 'flags':'-Wall -Wextra -O3 -fopenmp -std=c++17 -march=armv8-a+simd -shared -fPIC -lrknnrt',
 'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
 'scope':'public dynamic MatMul; configurable query tile; full K/V; scaled single or corrected two-pass probabilities'
},indent=2)+'\n')
PY
