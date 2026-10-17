#!/usr/bin/env bash
# Run on the already prepared RK3576 host; no image build or service replacement.
set -euo pipefail
lab=/opt/decision3-edge-latency
name=$1
script=$2
shift 2
env_args=()
port_args=()
if [[ -n ${D3_THREAD_LIMIT:-} ]]; then
  env_args+=(-e "OMP_THREAD_LIMIT=${D3_THREAD_LIMIT}" -e "OMP_NUM_THREADS=${D3_THREAD_LIMIT}")
fi
if [[ ${D3_TEMP_HTTP:-0} == 1 ]]; then
  port_args+=(-p 192.168.1.164:8901:8901)
fi
if [[ ${D3_OPENMP_PASSIVE:-0} == 1 ]]; then
  env_args+=(-e OMP_WAIT_POLICY=PASSIVE -e GOMP_SPINCOUNT=0)
fi
if [[ ${D3_OPENMP_ACTIVE:-0} == 1 ]]; then
  env_args+=(-e OMP_WAIT_POLICY=ACTIVE -e GOMP_SPINCOUNT=3000000)
fi
docker run --rm --name "decision3-latency-${name}" \
  --cpuset-cpus "${D3_CPUSET:-4-7}" --memory 8g --memory-swap 8g --device /dev/dri \
  "${env_args[@]}" "${port_args[@]}" \
  --mount type=bind,source=/opt/decision3-edge-matmul,target=/base,readonly \
  --mount "type=bind,source=${lab},target=/work" \
  --mount type=bind,source=/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime,target=/models,readonly \
  --workdir /work decision3-edge:arm64-cpu \
  "/work/${script}" \
  --model /models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e "$@"
