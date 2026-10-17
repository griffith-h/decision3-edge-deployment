#!/usr/bin/env bash
set -euo pipefail
lab=$(cd "$(dirname "$0")" && pwd)
for name in pipeline_bridge pipeline_profile; do
 g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -fopenmp -fno-semantic-interposition -ffp-contract=off -I /opt/decision3-edge-matmul/sdk "$lab/$name.cpp" -L /opt/decision3-edge-optimized-20261017-v1/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o "$lab/lib$name.so"
done
