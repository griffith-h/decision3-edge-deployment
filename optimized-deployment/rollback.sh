#!/usr/bin/env bash
# Stop the optimized release, then start the preserved original container.
set -euo pipefail
# Check the fallback exists before stopping the live service.
docker inspect decision3-edge-matmul >/dev/null
# Creation of the optimized container may itself have failed.
if docker inspect decision3-edge-optimized >/dev/null 2>&1; then
  docker stop -t 60 decision3-edge-optimized
fi
docker start decision3-edge-matmul
ready=0
for ((i=0; i<150; i++)); do
  if curl -fsS --max-time 5 http://192.168.1.164:8900/health; then
    ready=1
    break
  fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo 'Original service did not become ready; inspect docker logs decision3-edge-matmul' >&2; exit 1; }
