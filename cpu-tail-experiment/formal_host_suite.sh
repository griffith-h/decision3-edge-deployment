#!/usr/bin/env bash
set -euo pipefail
lab=/opt/decision3-edge-cpu-tail-20261017
cd "$lab"
bash run.sh select_runtime.py > logs/runtime-selection.log 2>&1
spin=$(python3 -c 'import json;print(json.load(open("runtime-selection.json"))["spin"] or "")')
D3_GOMP_SPINCOUNT="$spin" bash run.sh test_pipeline_guards.py > logs/pipeline-guards-final.log 2>&1
D3_GOMP_SPINCOUNT="$spin" bash run.sh final_benchmark.py --stage paired --count 20 > logs/formal-paired.log 2>&1
D3_GOMP_SPINCOUNT="$spin" bash run.sh final_benchmark.py --stage single --count 100 > logs/formal-single.log 2>&1
D3_GOMP_SPINCOUNT="$spin" bash run.sh profile_final.py > logs/final-profile.log 2>&1
