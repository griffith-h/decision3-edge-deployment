#!/usr/bin/env bash
set -euo pipefail
cd /opt/decision3-edge-npu
# Experimental artifact: probability parity FAILS; never promote to production.
model_file=${1:?Usage: bash run-board-test.sh path/to/experimental-model.rkllm}
args=()
for token_file in inputs/*.tokens; do
  hidden_file="output/$(basename "$token_file" .tokens).f32"
  args+=("$token_file" "$hidden_file")
done
: >logs/npu-load.log
timeout 240 ./npu_hidden "$model_file" "${args[@]}" >logs/npu-run.log 2>&1 &
npu_test_pid=$!
while kill -0 "$npu_test_pid" 2>/dev/null; do
  date -u '+%Y-%m-%dT%H:%M:%SZ' >>logs/npu-load.log
  cat /sys/kernel/debug/rknpu/load >>logs/npu-load.log
  sleep 0.25
done
wait "$npu_test_pid"
