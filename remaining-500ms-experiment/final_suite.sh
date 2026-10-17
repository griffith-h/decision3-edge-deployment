#!/usr/bin/env bash
set -euo pipefail
python -u /work/test_prepare.py > /work/logs/prepare-unit.log 2>&1
python -u /work/test_conv.py > /work/logs/conv-unit.log 2>&1
python -u /work/final_compare.py --variant conversion-conv-prepare --stage paired --count 20 > /work/logs/final-paired.log 2>&1
python -u /work/final_compare.py --variant conversion-conv-prepare --stage single --count 100 > /work/logs/final-single.log 2>&1
python -u /work/diagnose_candidate.py --variant conversion-conv-prepare > /work/logs/final-profile.log 2>&1
