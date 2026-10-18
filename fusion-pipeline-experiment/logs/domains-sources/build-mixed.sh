#!/usr/bin/env bash
set -euo pipefail
test -z "$(docker ps --filter name=decision3-fusion-pipeline-test --format '{{.Names}}')"
cd -- "$(dirname -- "$0")"
g++ -Wall -Wextra -O3 -std=c++17 -march=armv8-a+simd -fopenmp -shared -fPIC -I /opt/decision3-edge-matmul/sdk mixed_bridge.cpp -L /opt/decision3-edge-graph-20261018-v1/deps/release/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o libmixed_bridge.so
