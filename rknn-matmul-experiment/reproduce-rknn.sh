#!/usr/bin/env bash
# Run on the RK3576 board as root. Uses a separate directory for all new logs.
set -euo pipefail
source_dir=/opt/decision3-edge-matmul
model_cache=/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime
model_path=/models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e
run_dir=$(mktemp -d "$source_dir/reproduction-$(date +%Y%m%d-%H%M%S)-XXXXXX")
mkdir "$run_dir/logs"
for file in benchmark.py micro_test.py boundary_test.py cpu_optimized_server.py rknn_linear.py matmul_bridge.cpp request-cases.json golden-reference.json sdk-manifest.json; do
  cp "$source_dir/$file" "$run_dir/$file"
done
cp -a "$source_dir/sdk" "$run_dir/sdk"
python3 - "$run_dir" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / 'sdk-manifest.json').read_text())
for name, expected in manifest['files'].items():
    actual = hashlib.sha256((root / 'sdk' / name).read_bytes()).hexdigest()
    assert actual == expected['sha256'], f'SDK hash mismatch: {name}'
print('PASS: pinned RKNN SDK file hashes')
PY
restore_containers=()
restore_services() {
  for container in "${restore_containers[@]}"; do
    docker start "$container" || printf 'Failed to restore %s\n' "$container" >&2
  done
}
trap restore_services EXIT
for container in decision3-edge decision3-edge-matmul; do
  if [[ $(docker inspect --format '{{.State.Running}}' "$container" 2>/dev/null || true) == true ]]; then
    restore_containers+=("$container")
    docker stop "$container"
  fi
done
cd "$run_dir"
g++ -O2 -std=c++17 -shared -fPIC -I sdk matmul_bridge.cpp -L sdk -lrknnrt \
  -Wl,-rpath,'$ORIGIN/sdk' -o libmatmul_bridge.so
run_check() {
  docker run --rm --cpuset-cpus 4-7 --memory 8g --memory-swap 8g --device /dev/dri \
    --mount "type=bind,source=$model_cache,target=/models,readonly" \
    --mount "type=bind,source=$run_dir,target=/work" \
    --workdir /work decision3-edge:arm64-cpu "/work/$1" "${@:2}"
}
run_check micro_test.py 2>&1 | tee logs/micro-test.log
run_check boundary_test.py 2>&1 | tee logs/boundary-test.log
run_check benchmark.py --model "$model_path" --selection all --repeats 3 2>&1 | tee logs/benchmark-all.log
python3 - "$run_dir/logs/benchmark-all.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
assert report['passed']
for name, case in report['cases'].items():
    result = case['npu']
    print(f"{name}: CPU={case['cpu']['median_seconds']:.3f}s RKNN={result['median_seconds']:.3f}s "
          f"speedup={result['speedup']:.2f}x max_delta={result['max_probability_or_score_delta']:.8f} "
          f"same_choice={result['choices_unchanged']}")
PY
printf 'PASS; new evidence directory: %s/logs\n' "$run_dir"
