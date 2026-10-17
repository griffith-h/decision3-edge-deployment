#!/usr/bin/env bash
set -euo pipefail
python -u /work/test_small.py > /work/logs/small-unit-v2.log 2>&1
python -u /work/test_attention_guards.py > /work/logs/cpu-attention-guards.log 2>&1
python -u /work/test_rotary.py > /work/logs/rotary-unit.log 2>&1
python -u /work/test_residual.py > /work/logs/residual-unit.log 2>&1
python -u /work/test_bridge_lut.py > /work/logs/bridge-lut-unit.log 2>&1
python -u /work/test_bridge_modes.py > /work/logs/bridge-modes-unit.log 2>&1
python -u /work/screen_fusions.py > /work/logs/fusion-screen.log 2>&1
