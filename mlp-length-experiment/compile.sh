#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
docker run --rm --platform linux/amd64 --name decision3-mlp-bucket-compiler \
 --mount "type=bind,source=$repo/attention-chunk-experiment/compiler-env,target=/work/compiler-env,readonly" \
 --mount "type=bind,source=$repo/cpu-tail-experiment,target=/weights,readonly" \
 --mount "type=bind,source=$repo/mlp-length-experiment,target=/experiment" \
 --entrypoint /work/compiler-env/bin/python \
 sha256:abbed9aef590a8e2f565c695531e7ce57214ed626e30d5e14e317d0f4b0cfce3 -u /experiment/compile_buckets.py "$@"
