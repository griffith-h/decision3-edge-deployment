#!/usr/bin/env bash
# Run the benchmark or a resident JSON-lines worker in an isolated RK3576 container.
set -euo pipefail
lab=${1:?directory with src/ and results/}
model=${2:?checkpoint directory}
name=${3:?unique container name}
backend=${4:?cpu, rknn or infer}
shift 4
default_cpus=4-7
default_threads=3
default_blas=3
for option in "$@"; do
    if [[ $option == --aggressive ]]; then
        default_cpus=0-7
        default_threads=8
        default_blas=1
        D3_OMP_WAIT_POLICY=${D3_OMP_WAIT_POLICY:-PASSIVE}
        break
    fi
done
image=sha256:cd5c294eb741dd011ad31f8dbc6ab10a088123922122f06d2c29b7d89067d49d
options=(--network none)
pythonpath=/bench:/checkpoint
runner=(/bench/bench.py "$backend")
case "$backend" in
    cpu) ;;
    rknn|infer)
        options+=(--device /dev/dri -v "${D3_NATIVE:-$lab/native}:/native:ro"
                  -v "${D3_SDK:?directory containing librknnrt.so}:/sdk:ro" -e LD_LIBRARY_PATH=/sdk
                  --ulimit nofile=4096:4096)
        pythonpath+=:/native
        if [[ "$backend" == infer ]]; then
            options+=(-i)
            runner=(/bench/infer.py)
        fi
        ;;
    *) echo "unknown backend $backend" >&2; exit 2 ;;
esac
if [[ -n ${D3_OMP_WAIT_POLICY:-} ]]; then
    options+=(-e "OMP_WAIT_POLICY=$D3_OMP_WAIT_POLICY")
fi
# HF snapshots can link through repo-local blobs into a shared content-addressed store.
if [[ -d "$model/../../blobs" && "$(basename "$(dirname "$model")")" == snapshots ]]; then
    repo_cache=$(cd "$model/../.." && pwd)
    repo_name=$(basename "$repo_cache")
    checkpoint="/model-cache/$repo_name/snapshots/$(basename "$model")"
    model_mounts=(-v "$repo_cache:/model-cache/$repo_name:ro")
    shared_blobs="$(dirname "$repo_cache")/blobs"
    if [[ -d "$shared_blobs" ]]; then
        model_mounts+=(-v "$shared_blobs:/model-cache/blobs:ro")
    fi
else
    checkpoint=/checkpoint
    model_mounts=(-v "$model:/checkpoint:ro")
fi
docker run --rm --pull=never --name "$name" "${options[@]}" --cpuset-cpus "${D3_CPUS:-$default_cpus}" \
    --memory 8g --memory-swap 8g --ulimit core=0 --tmpfs /app/models \
    -e D3_THREADS="${D3_THREADS:-$default_threads}" -e OMP_NUM_THREADS="${D3_THREADS:-$default_threads}" \
    -e OPENBLAS_NUM_THREADS="${D3_BLAS_THREADS:-$default_blas}" -e MKL_NUM_THREADS="${D3_THREADS:-$default_threads}" \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e TOKENIZERS_PARALLELISM=false \
    -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH="$checkpoint:$pythonpath" -e D3_CHECKPOINT="$checkpoint" \
    "${model_mounts[@]}" -v "$lab/src:/bench:ro" -v "$lab/results:/out" \
    -w /bench --entrypoint python "$image" -u "${runner[@]}" "$@"
