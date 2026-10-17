#!/usr/bin/env bash
set -euo pipefail
cp /work/logs/scatter-balanced-none.json /work/logs/scatter-balanced-none-screen.json
python -u /work/scatter_runner.py paired.py --mode balanced --rounds 20 > /work/logs/final-native-paired-run.log 2>&1
python -u /work/scatter_runner.py bench.py --mode balanced --count 100 > /work/logs/final-native-100-run.log 2>&1
