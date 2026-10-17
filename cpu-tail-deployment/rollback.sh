#!/usr/bin/env bash
set -euo pipefail
# Preserve the actual previous three-thread container, including its restart policy.
docker inspect decision3-edge-optimized >/dev/null
if docker inspect decision3-edge-graph >/dev/null 2>&1; then
 docker update --restart=no decision3-edge-graph >/dev/null
 docker stop -t 90 decision3-edge-graph >/dev/null
fi
docker update --restart=unless-stopped decision3-edge-optimized >/dev/null
docker start decision3-edge-optimized >/dev/null
for ((i=0;i<150;i++)); do
 if curl -fsS --max-time 5 http://192.168.1.164:8900/health | python3 -c 'import json,sys; d=json.load(sys.stdin);assert d["deployment"]["release"]=="2026-10-17-three-thread-v1" and d["status"]=="ready"'; then exit 0; fi
 sleep 2
done
echo 'Rollback did not become ready; inspect decision3-edge-optimized logs' >&2
exit 1
