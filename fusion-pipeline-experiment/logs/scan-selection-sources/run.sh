#!/usr/bin/env bash
set -euo pipefail
lab=/opt/decision3-edge-fusion-pipeline-20261018
release=/opt/decision3-edge-graph-20261018-v1
args=()
for alias in references combined scan remainder projection previous attn; do
 args+=(--mount "type=bind,source=$release/deps/$alias,target=/$alias,readonly")
done
docker run --rm --init --name decision3-fusion-pipeline-test \
 --ulimit core=0 --ulimit nofile=4096:4096 --cpuset-cpus 4-7 --memory 8g --memory-swap 8g --device /dev/dri \
 -e OMP_NUM_THREADS=3 -e OMP_THREAD_LIMIT=3 -e OPENBLAS_NUM_THREADS=3 -e PYTHONPATH=/experiment:/tail:/long:/row:/mlp:/work \
 --mount "type=bind,source=$release/work,target=/work,readonly" \
 --mount "type=bind,source=$release/deps/release,target=/release,readonly" \
 --mount "type=bind,source=$release/deps/release/base,target=/base,readonly" \
 --mount type=bind,source=/opt/decision3-edge-long-tail-20261018,target=/tail,readonly \
 --mount type=bind,source=/opt/decision3-edge-long-sequence-20261018,target=/long,readonly \
 --mount type=bind,source=/opt/decision3-edge-row-projection-20261018,target=/row,readonly \
 --mount type=bind,source=/opt/decision3-edge-mlp-length-20261018,target=/mlp,readonly \
 --mount "type=bind,source=$lab,target=/experiment" \
 --mount type=bind,source=/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime,target=/models,readonly \
 "${args[@]}" --workdir /experiment --entrypoint python "$(cat "$release/image-id.txt")" -u "/experiment/$1" "${@:2}"
