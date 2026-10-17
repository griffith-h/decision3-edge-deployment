#!/usr/bin/env bash
set -euo pipefail
cd /opt/decision3-edge-projection-compile-20261017
if [[ -n $(docker ps --filter name=decision3-projection --format '{{.Names}}') ]]; then
 echo 'Stop/finish temporary tests before replacing a native library.' >&2
 exit 1
fi
g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -ffp-contract=off \
 -I /opt/decision3-edge-matmul/sdk safe_scatter_bridge.cpp \
 -L /opt/decision3-edge-optimized-20261017-v1/base/sdk -lrknnrt \
 -Wl,-rpath,/base/sdk -o libsafe_scatter_bridge.so
