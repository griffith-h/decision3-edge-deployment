#!/usr/bin/env bash
set -euo pipefail
python -u /work/test_lut.py > /work/logs/a72-lut-unit.log 2>&1
python -u /work/test_attention_prepare.py > /work/logs/a72-attention-prepare-unit.log 2>&1
python -u /work/test_pipeline.py > /work/logs/a72-pipeline-unit.log 2>&1
