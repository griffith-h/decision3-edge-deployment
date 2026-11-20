#!/usr/bin/env bash
set -euo pipefail
src=${1:?directory containing native_tiles.cpp}
edge=${2:?directory containing native_mlp.cpp}
sdk=${3:?RKNN SDK directory}
out=${4:?native library directory}
mkdir -p "$out"
# Keep the archived edge source intact. Its shape and layout checks also apply to these buckets.
python3 - "$edge/native_mlp.cpp" "$out/native_256.cpp" <<'PY'
import sys
from pathlib import Path
source = Path(sys.argv[1]).read_text()
for old, new in (("{128,256,512,640}", "{32,64,128,256}"), ("info.M=640;", "info.M=256;")):
    if source.count(old) != 1:
        raise ValueError(f"unexpected RKNN dispatch table: {old}")
    source = source.replace(old, new)
source = source.replace("num_threads(3)", "").replace("schedule(static)", "schedule(dynamic, 8)")
Path(sys.argv[2]).write_text(source)
PY
g++ -O3 -std=c++17 -march=armv8-a+simd -fopenmp -fPIC -shared -I "$sdk" \
    "$out/native_256.cpp" "$src/native_tiles.cpp" -L "$sdk" -lrknnrt -o "$out/libd3_tiles.so"
g++ -O3 -std=c++17 -march=armv8-a+simd -fopenmp -fPIC -shared -I "$sdk" -I "$edge" \
    "$src/prefix_segment.cpp" -L "$sdk" -lrknnrt -o "$out/libd3_segment.so"
