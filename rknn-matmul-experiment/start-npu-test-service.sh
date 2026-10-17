#!/usr/bin/env bash
set -euo pipefail
cd /opt/decision3-edge-matmul
# Never stop or replace decision3-edge. Re-running leaves an existing experiment alone.
if docker container inspect decision3-edge-matmul >/dev/null 2>&1; then
  printf '%s\n' 'Container decision3-edge-matmul already exists; inspect it before rerunning.' >&2
  exit 1
fi
model_cache=/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime
model_path=/models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e
docker run -d --name decision3-edge-matmul --restart unless-stopped --init \
  --cpuset-cpus 4-7 --memory 8g --memory-swap 8g --device /dev/dri \
  --log-opt max-size=10m --log-opt max-file=3 \
  -p 192.168.1.164:8900:8900 \
  --mount "type=bind,source=$model_cache,target=/models,readonly" \
  --mount type=bind,source=/opt/decision3-edge-matmul,target=/work,readonly \
  --workdir /work \
  --health-cmd 'python -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8900/health\", timeout=5).read()"' \
  --health-interval 30s --health-timeout 10s --health-start-period 180s \
  decision3-edge:arm64-cpu /work/npu_server.py --model "$model_path" --host 0.0.0.0 --port 8900
