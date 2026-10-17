#!/usr/bin/env bash
set -euo pipefail
cd /opt/decision3-edge-npu
g++ -O2 -std=c++17 -I runtime runtime/npu_hidden.cpp -L runtime -lrkllmrt \
  -Wl,-rpath,'$ORIGIN/runtime' -o npu_hidden
