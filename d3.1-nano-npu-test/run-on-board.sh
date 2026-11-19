#!/usr/bin/env bash
set -euo pipefail
lab=${1:?isolated experiment directory with native build and benchmark files}
cpu_lab=${2:?directory containing the CPU experiment model and request}
container=${3:?unique benchmark container name}
mode=${4:-default}
image_id=sha256:cd5c294eb741dd011ad31f8dbc6ab10a088123922122f06d2c29b7d89067d49d
extra=()
pythonpath=/bench:/native:/checkpoint
entry=run.py
hardware=(--device /dev/dri
          --mount "type=bind,src=$lab/adapter.py,dst=/bench/adapter.py,readonly"
          --mount "type=bind,src=$lab/native,dst=/native,readonly"
          --mount type=bind,src=/opt/decision3-edge-matmul/sdk,dst=/sdk,readonly
          --mount type=bind,src=/sys/kernel/debug/rknpu/load,dst=/rknpu-load,readonly)
case "$mode" in
    default) ;;
    single-question|single-question-cpu)
        release=/opt/decision3-edge-graph-20261018-v1
        for part in attn previous scan; do
            extra+=(--mount "type=bind,src=$release/deps/$part,dst=/$part,readonly")
        done
        extra+=(--mount "type=bind,src=$release/work,dst=/edge-work,readonly"
                --mount "type=bind,src=$release/deps/release/cpu_attention_optimized.py,dst=/scan-support/cpu_attention_optimized.py,readonly"
                -e D3_STATE_ONCE=0)
        pythonpath+=:/attn:/previous:/scan:/scan-support:/edge-work
        if [[ "$mode" == single-question-cpu ]]; then
            hardware=()
            entry=cpu-control.py
        fi
        ;;
    *) echo "Unknown benchmark mode $mode" >&2; exit 2 ;;
esac
mkdir -p "$lab/out"
docker run --pull=never --name "$container" --network none "${hardware[@]}" \
    --cpuset-cpus 4-7 --memory 8g --memory-swap 8g --ulimit core=0 --ulimit nofile=4096:4096 \
    -e OMP_NUM_THREADS=3 -e OPENBLAS_NUM_THREADS=3 -e MKL_NUM_THREADS=3 \
    -e TOKENIZERS_PARALLELISM=false -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$pythonpath" -e LD_LIBRARY_PATH=/sdk "${extra[@]}" \
    --mount "type=bind,src=$cpu_lab/model,dst=/checkpoint,readonly" \
    --mount "type=bind,src=$cpu_lab/request.json,dst=/bench/request.json,readonly" \
    --mount "type=bind,src=$cpu_lab/benchmark.py,dst=/bench/cpu_probe.py,readonly" \
    --mount "type=bind,src=$lab/$entry,dst=/bench/$entry,readonly" \
    --mount "type=bind,src=$lab/out,dst=/out" --tmpfs /app/models --workdir /bench \
    --entrypoint python "$image_id" -u "/bench/$entry"
