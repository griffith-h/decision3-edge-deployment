#!/usr/bin/env bash
set -euo pipefail
lab=/opt/decision3-edge-bridge-parallel-20261017
release=/opt/decision3-edge-optimized-20261017-v1
cd "$lab"
mkdir -p logs
cc -O2 -fPIC -shared -fopenmp cpu_control.c -o libcpu_control.so
image=$(cat "$release/image-id.txt")
docker run --rm --init --name decision3-bridge-shadow-check \
  --ulimit nofile=4096:4096 --ulimit core=0 --cpuset-cpus 4-7 --memory 8g --memory-swap 8g --device /dev/dri \
  -e OMP_NUM_THREADS=3 -e OMP_THREAD_LIMIT=3 -e OPENBLAS_NUM_THREADS=3 \
  --mount "type=bind,source=$release,target=/release,readonly" \
  --mount "type=bind,source=$release/base,target=/base,readonly" \
  --mount type=bind,source=/opt/decision3-edge-matmul,target=/references,readonly \
  --mount "type=bind,source=$lab,target=/work" \
  --mount type=bind,source=/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime,target=/models,readonly \
  --workdir /work "$image" /work/shadow_check.py \
  --model /models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e "$@"
