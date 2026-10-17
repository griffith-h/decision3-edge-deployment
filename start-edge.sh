#!/usr/bin/env bash
set -euo pipefail
cd /opt/decision3-edge
if docker container inspect decision3-edge >/dev/null 2>&1; then
  docker rm -f decision3-edge >/dev/null
fi
model_cache=/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime
model_path=/models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e
docker run -d --name decision3-edge --restart unless-stopped --init \
  --cpuset-cpus 4-7 \
  --memory 8g --memory-swap 8g \
  --log-opt max-size=10m --log-opt max-file=3 \
  -p 192.168.1.164:8899:8899 \
  --mount "type=bind,source=$model_cache,target=/models,readonly" \
  --mount type=bind,source=/opt/decision3-edge/cpu_optimized_server.py,target=/app/cpu_optimized_server.py,readonly \
  --health-cmd 'python -c "import urllib.request; urllib.request.urlopen(\"http://127.0.0.1:8899/health\", timeout=5).read()"' \
  --health-interval 30s --health-timeout 10s --health-start-period 180s \
  decision3-edge:arm64-cpu /app/cpu_optimized_server.py \
  --model "$model_path" --host 0.0.0.0 --port 8899
