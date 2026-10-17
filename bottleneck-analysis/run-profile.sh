#!/usr/bin/env bash
set -euo pipefail
lab=/opt/decision3-edge-bottlenecks-20261017
release=/opt/decision3-edge-optimized-20261017-v1
cd "$lab"
mkdir -p logs
g++ -Wall -Wextra -O2 -std=c++17 -shared -fPIC \
  -I /opt/decision3-edge-matmul/sdk matmul_profile.cpp \
  -L "$release/base/sdk" -lrknnrt -Wl,-rpath,/base/sdk -o libmatmul_profile.so
image=$(cat "$release/image-id.txt")
docker run --rm --init --name decision3-bottleneck-profile \
  --cpuset-cpus 4-7 --memory 8g --memory-swap 8g --device /dev/dri \
  -e OMP_NUM_THREADS=3 -e OMP_THREAD_LIMIT=3 -e OPENBLAS_NUM_THREADS=3 \
  --mount "type=bind,source=$release,target=/release,readonly" \
  --mount "type=bind,source=$release/base,target=/base,readonly" \
  --mount "type=bind,source=$lab,target=/work" \
  --mount type=bind,source=/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime,target=/models,readonly \
  --workdir /work "$image" /work/profile_current.py \
  --model /models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e
