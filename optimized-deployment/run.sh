#!/usr/bin/env bash
# Run on the prepared RK3576 host. This never stops or removes another container.
set -euo pipefail
release_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
name=${1:-decision3-edge-optimized}
port=${2:-8900}
restart=${3:-unless-stopped}
bind_ip=${D3_BIND_IP:-192.168.1.164}
model_root=${D3_MODEL_ROOT:-/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime}
image=$(cat "$release_dir/image-id.txt")
[[ -f "$release_dir/libd3_native_norm.so" && -f "$release_dir/base/libmatmul_bridge.so" ]]
docker run -d --name "$name" --init --restart "$restart" \
  --cpuset-cpus 4-7 --memory 8g --memory-swap 8g --device /dev/dri \
  -e OMP_NUM_THREADS=3 -e OMP_THREAD_LIMIT=3 -e OPENBLAS_NUM_THREADS=3 \
  -p "$bind_ip:$port:8900" \
  --mount "type=bind,source=$release_dir,target=/work,readonly" \
  --mount "type=bind,source=$release_dir/base,target=/base,readonly" \
  --mount "type=bind,source=$model_root,target=/models,readonly" \
  --workdir /work --log-opt max-size=10m --log-opt max-file=3 \
  --health-cmd 'curl --fail --silent --show-error --max-time 5 --output /dev/null http://127.0.0.1:8900/health' \
  --health-interval 30s --health-timeout 10s --health-start-period 180s \
  "$image" /work/server.py --model \
  /models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e --port 8900
