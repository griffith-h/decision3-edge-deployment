#!/usr/bin/env bash
set -euo pipefail
lab=$(cd "$(dirname "$0")" && pwd)
g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -I /opt/decision3-edge-matmul/sdk "$lab/mlp_graph_runtime.cpp" -L /opt/decision3-edge-optimized-20261017-v1/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o "$lab/libmlp_graph_runtime.so"
