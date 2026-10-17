#!/usr/bin/env bash
set -euo pipefail
lab=$(cd "$(dirname "$0")" && pwd)
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -ffp-contract=off "$lab/lut_math.c" -lm -o "$lab/liblut_math.so"
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -ffp-contract=off "$lab/cpu_attention.c" -lm -o "$lab/libcpu_attention.so"
g++ -Wall -Wextra -O3 -std=c++17 -fPIC -shared -ffp-contract=off -I /opt/decision3-edge-matmul/sdk "$lab/lut_bridge.cpp" -L /opt/decision3-edge-optimized-20261017-v1/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o "$lab/liblut_bridge.so"
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -ffp-contract=off "$lab/rotary.c" -o "$lab/librotary.so"
for name in native_bridge m115_bridge; do
 g++ -Wall -Wextra -O3 -std=c++17 -fPIC -shared -fno-semantic-interposition -ffp-contract=off -I /opt/decision3-edge-matmul/sdk "$lab/$name.cpp" -L /opt/decision3-edge-optimized-20261017-v1/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o "$lab/lib$name.so"
done
