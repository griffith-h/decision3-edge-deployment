#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
test -z "$(docker ps --filter name=decision3-bf16-pipeline-test --format '{{.Names}}')"
gcc -Wall -Wextra -O3 -march=armv8-a+simd -fopenmp -shared -fPIC stream_scan.c -o libstream_scan.so
# Immutable fifth-stage binaries are reused via read-only mount /fusion.
for f in libmixed_bridge.so libscan_four.so libdynamic_matmul.so; do
 ln -snf "/fusion/$f" "$f"
done
