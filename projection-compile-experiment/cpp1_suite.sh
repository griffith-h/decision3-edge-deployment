#!/usr/bin/env bash
set -euo pipefail
python -u /work/probe_groups.py > /work/logs/group-probe-run.log 2>&1
python -u /work/static_runner.py scatter_runner.py bench.py --mode balanced --compile all --count 16 > /work/logs/scatter-balanced-static-all-run.log 2>&1
cp /work/logs/baseline-all-static.json /work/logs/baseline-all-static-cpp3.json
cp /work/logs/scatter-balanced-all-static.json /work/logs/scatter-balanced-all-static-cpp3.json
python -u /work/cpp1_runner.py static_runner.py bench.py --mode baseline --compile all --count 16 > /work/logs/compile-cpp1-all-run.log 2>&1
python -u /work/cpp1_runner.py static_runner.py scatter_runner.py bench.py --mode balanced --compile all --count 16 > /work/logs/scatter-balanced-cpp1-all-run.log 2>&1
