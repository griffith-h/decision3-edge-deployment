#!/usr/bin/env bash
set -euo pipefail
lab=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$lab/a72"
for name in lut_math rotary cpu_attention; do
 gcc -Wall -Wextra -O3 -mcpu=cortex-a72 -mtune=cortex-a72 -shared -fPIC -fopenmp -ffp-contract=off "$lab/$name.c" -lm -o "$lab/a72/lib$name.so"
done
gcc -Wall -Wextra -O3 -mcpu=cortex-a72 -mtune=cortex-a72 -shared -fPIC -fopenmp -ffp-contract=off /opt/decision3-edge-attention-chunk-20261017/chunk_blas2.c -o "$lab/a72/libchunk_blas2.so"
for name in pipeline_bridge pipeline_profile; do
 g++ -Wall -Wextra -O3 -std=c++17 -mcpu=cortex-a72 -mtune=cortex-a72 -shared -fPIC -fopenmp -ffp-contract=off -fno-semantic-interposition -I /opt/decision3-edge-matmul/sdk "$lab/$name.cpp" -L /opt/decision3-edge-optimized-20261017-v1/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o "$lab/a72/lib$name.so"
done
