#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
test -z "$(docker ps --filter name=decision3-bf16-pipeline-test --format '{{.Names}}')"
for f in libmixed_bridge.so libscan_four.so libdynamic_matmul.so; do ln -snf "/fusion/$f" "$f"; done
bash build-native.sh
bash build-native-split.sh
