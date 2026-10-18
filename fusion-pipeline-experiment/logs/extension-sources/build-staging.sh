#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -ffp-contract=off -fopenmp -I /opt/decision3-edge-matmul/sdk staging_bridge.cpp -L /opt/decision3-edge-graph-20261018-v1/deps/release/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o libstaging_bridge.so
