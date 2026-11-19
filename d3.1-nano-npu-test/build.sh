#!/usr/bin/env bash
set -euo pipefail
# Build the archived edge kernels with four short sequence buckets for nano.
edge=${1:?edge native_mlp source directory}
sdk=${2:?RKNN header and library directory}
out=${3:?isolated build directory}
mkdir -p "$out"
python3 - "$edge/native_mlp.cpp" "$out/native_short.cpp" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text()
for old, new in (("{128,256,512,640}", "{32,64,96,128}"), ("info.M=640;", "info.M=128;")):
    if source.count(old) != 1:
        raise ValueError(f"Unexpected edge dispatch table in {sys.argv[1]}")
    source = source.replace(old, new)
Path(sys.argv[2]).write_text(source)
PY
cp "$edge/native_mlp.py" "$out/native_mlp.py"
g++ -O3 -std=c++17 -march=armv8-a+simd -fopenmp -fPIC -shared \
    -I "$sdk" "$out/native_short.cpp" -L "$sdk" -lrknnrt -o "$out/libnative_short.so"
