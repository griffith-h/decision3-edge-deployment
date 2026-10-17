#!/usr/bin/env bash
set -euo pipefail
cd /opt/decision3-edge-matmul
selection=${1:-all}
model_cache=/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime
model_path=/models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e
docker run --rm --name decision3-matmul-benchmark --cpuset-cpus 4-7 \
  --memory 8g --memory-swap 8g --device /dev/dri \
  --mount "type=bind,source=$model_cache,target=/models,readonly" \
  --mount type=bind,source=/opt/decision3-edge-matmul,target=/work \
  --workdir /work decision3-edge:arm64-cpu /work/benchmark.py \
  --model "$model_path" --selection "$selection" --repeats 3
