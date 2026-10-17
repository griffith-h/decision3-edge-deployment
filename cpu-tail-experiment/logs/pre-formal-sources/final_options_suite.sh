#!/usr/bin/env bash
set -euo pipefail
python -u /work/test_mlp_io.py > /work/logs/mlp-half-io-unit.log 2>&1
python -u /work/screen_final_options.py > /work/logs/final-options-screen.log 2>&1
