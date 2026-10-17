#!/usr/bin/env bash
set -euo pipefail
python -u /work/test_mlp_graph_guards.py > /work/logs/mlp-graph-guards.log 2>&1
python -u /work/screen_mlp_graph.py > /work/logs/mlp-graph-screen.log 2>&1
