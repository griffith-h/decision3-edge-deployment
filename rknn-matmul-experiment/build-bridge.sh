#!/usr/bin/env bash
set -euo pipefail
cd /opt/decision3-edge-matmul
g++ -O2 -std=c++17 -shared -fPIC -I sdk matmul_bridge.cpp -L sdk -lrknnrt \
  -Wl,-rpath,'$ORIGIN/sdk' -o libmatmul_bridge.so
