#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
test -z "$(docker ps --filter name=decision3-bf16-pipeline-test --format '{{.Names}}')"
g++ -Wall -Wextra -O3 -std=c++17 -march=armv8-a+simd -fopenmp -shared -fPIC -I /opt/decision3-edge-matmul/sdk fused_boundary.cpp -L /opt/decision3-edge-graph-20261018-v1/deps/release/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o libfused_boundary.so
