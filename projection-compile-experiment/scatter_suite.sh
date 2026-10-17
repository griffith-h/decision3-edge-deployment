#!/usr/bin/env bash
set -euo pipefail
python -u /work/static_runner.py test_compile.py > /work/logs/compile-static-unit-run.log 2>&1
cp /work/logs/compile-unit.json /work/logs/compile-static-unit.json
python -u /work/scatter_runner.py test_packing.py > /work/logs/scatter-unit-run.log 2>&1
cp /work/logs/baseline-none.json /work/logs/pre-compiler-baseline-none.json
python -u /work/bench.py --mode baseline --count 16 > /work/logs/compiler-baseline-run.log 2>&1
cp /work/logs/baseline-none.json /work/logs/compiler-baseline-none.json
python -u /work/static_runner.py bench.py --mode baseline --compile all --count 16 > /work/logs/compile-static-all-run.log 2>&1
python -u /work/scatter_runner.py bench.py --mode merge --count 16 > /work/logs/scatter-merge-run.log 2>&1
python -u /work/scatter_runner.py bench.py --mode balanced --count 16 > /work/logs/scatter-balanced-run.log 2>&1
python -u /work/probe_groups.py > /work/logs/group-probe-run.log 2>&1
# Dynamic CPU variants were slower. Test the full combined static candidate.
python -u /work/static_runner.py scatter_runner.py bench.py --mode balanced --compile all --count 16 > /work/logs/scatter-balanced-static-all-run.log 2>&1
