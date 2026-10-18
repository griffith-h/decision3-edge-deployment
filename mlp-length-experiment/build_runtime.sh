#!/usr/bin/env bash
set -euo pipefail
cd /opt/decision3-edge-mlp-length-20261018
g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -I /opt/decision3-edge-matmul/sdk -I /opt/decision3-edge-graph-20261018-v1/work row_graph_runtime.cpp -L /opt/decision3-edge-graph-20261018-v1/deps/release/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o librow_graph_runtime.so
