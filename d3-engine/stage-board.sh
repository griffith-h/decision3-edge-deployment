#!/usr/bin/env bash
# Run from a checkout on the RK3576 with the RKNN MatMul SDK installed.
set -euo pipefail
lab=${1:?isolated build directory}
sdk=${2:?directory with rknn_matmul_api.h and librknnrt.so}
src=$(cd "$(dirname "$0")" && pwd)
repo=$(dirname "$src")
mkdir -p "$lab/src" "$lab/results"
cp "$src"/{bench.py,compare.py,cpu_attention.py,engine.py,infer.py,state_kernels.py,test_engine.py} "$lab/src/"
cp "$src"/{state_scan.c,build-state.sh,run-board.sh} "$lab/src/"
cp "$repo/d3.1-nano-npu-test/adapter.py" "$repo/remaining-500ms-experiment/"{conv_btc.c,scan_prepare.c} "$lab/src/"
bash "$repo/d3.1-nano-npu-test/build.sh" "$repo/bf16-pipeline-experiment" "$sdk" "$lab/native"
bash "$lab/src/build-state.sh" "$lab/src" "$lab/src/libd3_state.so"
