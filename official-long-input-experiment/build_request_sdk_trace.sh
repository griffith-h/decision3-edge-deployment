#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
g++ -Wall -Wextra -O2 -std=c++17 -shared -fPIC \
 -I /opt/decision3-edge-matmul/sdk request_sdk_trace.cpp \
 -ldl -pthread -o librequest_sdk_trace.so
python3 - <<'PY'
import hashlib,json,subprocess
from pathlib import Path
paths=['request_sdk_trace.cpp','build_request_sdk_trace.sh','librequest_sdk_trace.so']
Path('request-sdk-trace-build.json').write_text(json.dumps({
 'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
 'sdk_header_sha256':hashlib.sha256(Path('/opt/decision3-edge-matmul/sdk/rknn_matmul_api.h').read_bytes()).hexdigest(),
 'sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths},
 'scope':'Diagnostic public rknn_matmul_run pass-through; host synchronous duration, not pure device time'},indent=2)+'\n')
PY
