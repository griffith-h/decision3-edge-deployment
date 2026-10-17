#!/usr/bin/env bash
set -euo pipefail
lab=/opt/decision3-edge-cpu-tail-20261017
cd "$lab"
if [[ -e gpu-worker.sock ]]; then echo 'Check existing worker before rerun' >&2; exit 1; fi
taskset -c 0-3 bash gpu_namespace.sh python3 -u gpu_worker.py > logs/gpu-worker.log 2>&1 &
worker_pid=$!
cleanup(){ python3 stop_gpu.py || true; wait "$worker_pid" || true; }
trap cleanup EXIT
for i in $(seq 1 60); do
 if [[ -S gpu-worker.sock ]]; then break; fi
 if ! kill -0 "$worker_pid" 2>/dev/null; then cat logs/gpu-worker.log; exit 1; fi
 sleep 1
done
[[ -S gpu-worker.sock ]]
bash run.sh test_gpu.py > logs/gpu-unit.log 2>&1
bash run.sh screen_gpu.py > logs/gpu-screen.log 2>&1
