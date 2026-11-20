#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if test -e native_split_reference.inc; then
 cmp -s /opt/decision3-edge-bf16-pipeline-20261018/native_split.cpp native_split_reference.inc
else
 cp /opt/decision3-edge-bf16-pipeline-20261018/native_split.cpp native_split_reference.inc
fi
g++ -Wall -Wextra -O3 -fopenmp -std=c++17 -march=armv8-a+simd -shared -fPIC \
 -I /opt/decision3-edge-matmul/sdk linear_precision_bridge.cpp \
 -L /opt/decision3-edge-graph-20261018-v1/deps/release/base/sdk \
 -lrknnrt -Wl,-rpath,/base/sdk -o liblinear_precision_probe.so
python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
files=['linear_precision_bridge.cpp','native_split_reference.inc','build_linear_precision.sh','liblinear_precision_probe.so']
Path('linear-precision-build.json').write_text(json.dumps({'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
 'scope':'Public SDK FP16xFP16-to-FP32, identical real first-layer input; prototype native layout; no full-model performance claim',
 'sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}},indent=2)+'\n')
PY
