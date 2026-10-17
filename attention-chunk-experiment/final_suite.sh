#!/usr/bin/env bash
set -euo pipefail
python -u /work/test_graph_guards.py > /work/logs/graph-guard-unit.log 2>&1
python -u /work/test_chunk_wrapper.py > /work/logs/chunk-wrapper-unit.log 2>&1
python -u /work/final_eval.py --stage validate > /work/logs/final-validate.log 2>&1
python -c 'import json;d=json.load(open("/work/logs/final-validate-combined.json"));assert d["variants"]["combined"]["passed"],"combined candidate failed numerical validation"'
python -u /work/final_eval.py --stage paired --variant combined --count 20 > /work/logs/final-paired.log 2>&1
python -u /work/final_eval.py --stage single --variant combined --count 100 > /work/logs/final-single.log 2>&1
python -u /work/read_graph_profile.py > /work/logs/graph-profile.log 2>&1
