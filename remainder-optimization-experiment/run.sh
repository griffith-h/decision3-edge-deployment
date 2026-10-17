#!/usr/bin/env bash
set -euo pipefail
lab=/opt/decision3-edge-remainder-20261017
release=/opt/decision3-edge-optimized-20261017-v1
cd "$lab"
mkdir -p logs
if docker container inspect decision3-remainder-test >/dev/null 2>&1; then
  echo 'Experiment container already exists; refusing to overwrite its library.' >&2
  exit 1
fi
g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -ffp-contract=off \
 -I /opt/decision3-edge-matmul/sdk fused_profile.cpp -L "$release/base/sdk" \
 -lrknnrt -Wl,-rpath,/base/sdk -o libfused_profile.so
gcc -Wall -Wextra -O3 -std=c11 -shared -fPIC -ffp-contract=off -fopenmp cpu_math.c -o libcpu_math.so -lm
g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -ffp-contract=off \
 -I /opt/decision3-edge-matmul/sdk half_bridge.cpp -L "$release/base/sdk" \
 -lrknnrt -Wl,-rpath,/base/sdk -o libhalf_bridge.so
g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -ffp-contract=off \
 -I /opt/decision3-edge-matmul/sdk half_profile.cpp -L "$release/base/sdk" \
 -lrknnrt -Wl,-rpath,/base/sdk -o libhalf_profile.so
gcc -Wall -Wextra -O3 -std=c11 -shared -fPIC -ffp-contract=off -fopenmp cpu_math_sleef.c -o libcpu_math_sleef.so -lm
g++ -Wall -Wextra -O3 -std=c++17 -shared -fPIC -ffp-contract=off \
 -I /opt/decision3-edge-matmul/sdk safe_half_bridge.cpp -L "$release/base/sdk" \
 -lrknnrt -Wl,-rpath,/base/sdk -o libsafe_half_bridge.so
image=$(cat "$release/image-id.txt")
extra_env=()
if [[ ${D3_WAIT_POLICY:-} == passive ]]; then
  extra_env=(-e OMP_WAIT_POLICY=PASSIVE -e GOMP_SPINCOUNT=0)
fi
docker run --rm --init --name decision3-remainder-test \
  --ulimit core=0 --ulimit nofile=4096:4096 --cpuset-cpus 4-7 --memory 8g --memory-swap 8g --device /dev/dri \
  -e OMP_NUM_THREADS=3 -e OMP_THREAD_LIMIT=3 -e OPENBLAS_NUM_THREADS=3 "${extra_env[@]}" \
  --mount "type=bind,source=$release,target=/release,readonly" \
  --mount "type=bind,source=$release/base,target=/base,readonly" \
  --mount type=bind,source=/opt/decision3-edge-matmul,target=/references,readonly \
  --mount type=bind,source=/opt/decision3-edge-bridge-parallel-20261017,target=/combined,readonly \
  --mount type=bind,source=/opt/decision3-edge-scan-20261017,target=/scan,readonly \
  --mount "type=bind,source=$lab,target=/work" \
  --mount type=bind,source=/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime,target=/models,readonly \
  --workdir /work "$image" "/work/${D3_REMAINDER_SCRIPT:-profile_remainder.py}" \
  --model /models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e "$@"
