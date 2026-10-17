#!/usr/bin/env bash
# Run on the RK3576 host, never inside production. Output only in this lab.
set -euo pipefail
lab=$(cd "$(dirname "$0")" && pwd)
for name in conv_btc scan_prepare scan_fast; do
 gcc -Wall -Wextra -O3 -shared -fPIC -fopenmp -ffp-contract=off "$lab/$name.c" -lm -o "$lab/lib$name.so"
done
for name in profile_scatter cacheable_bridge conversion_bridge profile_conversion; do
 g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -ffp-contract=off \
  -I /opt/decision3-edge-matmul/sdk "$lab/$name.cpp" \
  -L /opt/decision3-edge-optimized-20261017-v1/base/sdk -lrknnrt \
  -Wl,-rpath,/base/sdk -o "$lab/lib$name.so"
done
