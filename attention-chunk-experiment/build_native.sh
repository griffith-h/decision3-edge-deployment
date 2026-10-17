#!/usr/bin/env bash
# Run only on RK3576, only after previous test processes have exited.
set -euo pipefail
lab=$(cd "$(dirname "$0")" && pwd)
for name in chunk_scan chunk_scan_v2 chunk_blas chunk_blas2; do
 gcc -Wall -Wextra -O3 -shared -fPIC -fopenmp -ffp-contract=off "$lab/$name.c" -o "$lab/lib$name.so"
done
for name in graph_runtime graph_profile core_runtime core_profile core_diagnostic core_nchw; do
 g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -I /opt/decision3-edge-matmul/sdk "$lab/$name.cpp" \
  -L /opt/decision3-edge-optimized-20261017-v1/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o "$lab/lib$name.so"
done
