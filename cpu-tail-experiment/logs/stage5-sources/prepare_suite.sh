#!/usr/bin/env bash
set -euo pipefail
python -u /work/test_attention_prepare.py > /work/logs/attention-prepare-unit.log 2>&1
python -u /work/screen_prepare.py > /work/logs/prepare-screen.log 2>&1
