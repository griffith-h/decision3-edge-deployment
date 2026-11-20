#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
test ! -e mixed_bridge_reference.inc
cp /opt/decision3-edge-bf16-pipeline-20261018/mixed_bridge.cpp mixed_bridge_reference.inc
g++ -Wall -Wextra -O3 -fopenmp -std=c++17 -march=armv8-a+simd -shared -fPIC \
 -I /opt/decision3-edge-matmul/sdk projection_read_bridge.cpp \
 -L /opt/decision3-edge-graph-20261018-v1/deps/release/base/sdk \
 -lrknnrt -Wl,-rpath,/base/sdk -o libprojection_read_probe.so
python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
files=['projection_read_bridge.cpp','mixed_bridge_reference.inc','build_projection_read.sh','libprojection_read_probe.so']
Path('projection-read-build.json').write_text(json.dumps({'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
 'scope':'Copy completed own normal-layout FP32 context only; Owner retains all handles',
 'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}},indent=2)+'\n')
PY
