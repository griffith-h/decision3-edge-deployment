#!/usr/bin/env bash
set -euo pipefail
python -u /work/qualified_benchmark.py --stage paired --count 20 > /work/logs/qualified-paired.log 2>&1
selected=$(python -c 'import json;print(json.load(open("/work/logs/qualified-paired.json"))["selected"])')
python -u /work/core_eval.py --stage single --variant "$selected" --count 100 > /work/logs/qualified-single.log 2>&1
