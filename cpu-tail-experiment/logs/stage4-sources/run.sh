#!/usr/bin/env bash
set -euo pipefail
lab=/opt/decision3-edge-cpu-tail-20261017
release=/opt/decision3-edge-optimized-20261017-v1
image=$(cat "$release/image-id.txt")
extra_env=()
if [[ -n ${D3_GOMP_SPINCOUNT:-} ]]; then extra_env+=(-e "GOMP_SPINCOUNT=$D3_GOMP_SPINCOUNT"); fi
docker run --rm --init --name decision3-cputail-test \
 --ulimit core=0 --ulimit nofile=4096:4096 --cpuset-cpus 4-7 --memory 8g --memory-swap 8g --device /dev/dri \
 -e OMP_NUM_THREADS=3 -e "OMP_THREAD_LIMIT=${D3_THREAD_LIMIT:-3}" -e OPENBLAS_NUM_THREADS=3 "${extra_env[@]}" \
 --mount "type=bind,source=$release,target=/release,readonly" \
 --mount "type=bind,source=$release/base,target=/base,readonly" \
 --mount type=bind,source=/opt/decision3-edge-matmul,target=/references,readonly \
 --mount type=bind,source=/opt/decision3-edge-bridge-parallel-20261017,target=/combined,readonly \
 --mount type=bind,source=/opt/decision3-edge-scan-20261017,target=/scan,readonly \
 --mount type=bind,source=/opt/decision3-edge-remainder-20261017,target=/remainder,readonly \
 --mount type=bind,source=/opt/decision3-edge-projection-compile-20261017,target=/projection,readonly \
 --mount type=bind,source=/opt/decision3-edge-remaining-500ms-20261017,target=/previous,readonly \
 --mount type=bind,source=/opt/decision3-edge-attention-chunk-20261017,target=/attn,readonly \
 --mount "type=bind,source=$lab,target=/work" \
 --mount type=bind,source=/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime,target=/models,readonly \
 --workdir /work --entrypoint /bin/bash "$image" /work/entry.sh "$@"
