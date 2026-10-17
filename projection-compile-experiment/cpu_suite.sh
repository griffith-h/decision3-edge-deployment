#!/usr/bin/env bash
set -euo pipefail
python -u /work/test_compile.py > /work/logs/compile-unit-run.log 2>&1
python -u /work/bench.py --mode baseline --compile mlp --count 16 > /work/logs/compile-mlp-run.log 2>&1
python -u /work/bench.py --mode baseline --compile all --count 16 > /work/logs/compile-all-run.log 2>&1
