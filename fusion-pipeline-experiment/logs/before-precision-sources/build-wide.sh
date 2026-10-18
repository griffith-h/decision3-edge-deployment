#!/usr/bin/env bash
set -euo pipefail
if [[ -n $(docker ps --filter name=decision3-fusion-pipeline-test --format '{{.Names}}') ]]; then
 echo 'Finish the isolated test before replacing its libraries.' >&2
 exit 1
fi
cd -- "$(dirname -- "$0")"
g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -ffp-contract=off -fopenmp -I /opt/decision3-edge-matmul/sdk wide_bridge.cpp -L /opt/decision3-edge-graph-20261018-v1/deps/release/base/sdk -lrknnrt -Wl,-rpath,/base/sdk -o libwide_bridge.so
