#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd attention.c -lm -o libattention.so
g++ -Wall -Wextra -O3 -std=c++17 -march=armv8-a+simd -shared -fPIC -I /opt/decision3-edge-matmul/sdk dynamic_matmul.cpp -L /opt/decision3-edge-graph-20261018-v1/deps/release/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o libdynamic_matmul.so
