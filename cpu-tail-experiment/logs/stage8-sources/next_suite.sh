#!/usr/bin/env bash
set -euo pipefail
python -u /work/test_lut.py > /work/logs/lut-unit-layout.log 2>&1
python -u /work/activation_replay.py > /work/logs/activation-replay.log 2>&1
python -u /work/test_small.py > /work/logs/small-unit.log 2>&1
python -u /work/test_attention.py > /work/logs/cpu-attention-unit.log 2>&1
python -u /work/screen_tail.py > /work/logs/tail-screen.log 2>&1
