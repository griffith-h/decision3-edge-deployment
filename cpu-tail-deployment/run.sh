#!/usr/bin/env bash
# Launch only; does not stop, rename, or remove any existing container.
set -euo pipefail
release_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
name=${1:-decision3-edge-graph}
port=${2:-8900}
restart=${3:-unless-stopped}
bind_ip=${D3_BIND_IP:-192.168.1.164}
model_root=${D3_MODEL_ROOT:-/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime}
image=$(cat "$release_dir/image-id.txt")
(cd "$release_dir" && sha256sum --quiet -c SHA256SUMS)
args=()
for alias in references combined scan remainder projection previous attn; do
 args+=(--mount "type=bind,source=$release_dir/deps/$alias,target=/$alias,readonly")
done
docker run -d --name "$name" --init --restart "$restart" \
 --ulimit core=0 --ulimit nofile=4096:4096 --cpuset-cpus 4-7 --memory 8g --memory-swap 8g --device /dev/dri \
 -e OMP_NUM_THREADS=3 -e OMP_THREAD_LIMIT=3 -e OPENBLAS_NUM_THREADS=3 \
 -p "$bind_ip:$port:8900" \
 --mount "type=bind,source=$release_dir/work,target=/work,readonly" \
 --mount "type=bind,source=$release_dir/service,target=/service,readonly" \
 --mount "type=bind,source=$release_dir/deps/release,target=/release,readonly" \
 --mount "type=bind,source=$release_dir/deps/release/base,target=/base,readonly" \
 --mount "type=bind,source=$model_root,target=/models,readonly" "${args[@]}" \
 --workdir /service --log-opt max-size=10m --log-opt max-file=3 \
 --health-cmd 'curl --fail --silent --show-error --max-time 5 --output /dev/null http://127.0.0.1:8900/health' \
 --health-interval 30s --health-timeout 10s --health-start-period 240s \
 "$image" /service/server.py --model \
 /models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e --port 8900
