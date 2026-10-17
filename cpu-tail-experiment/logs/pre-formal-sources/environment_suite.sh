#!/usr/bin/env bash
# Host orchestration; every Python run is a separate --rm test container.
set -euo pipefail
lab=/opt/decision3-edge-cpu-tail-20261017
cd "$lab"
taskset -c 0-3 bash build_mlp_graph.sh > logs/build-mlp-cores.log 2>&1
bash run.sh probe_mlp_cores.py > logs/mlp-core-probe.log 2>&1
bash run.sh screen_mlp_multicore.py > logs/mlp-multicore-screen.log 2>&1
bash run.sh env_probe.py --tag default-before > logs/environment-default-before.log 2>&1
D3_GOMP_SPINCOUNT=0 bash run.sh env_probe.py --tag spin0 > logs/environment-spin0.log 2>&1
D3_GOMP_SPINCOUNT=1000 bash run.sh env_probe.py --tag spin1000 > logs/environment-spin1000.log 2>&1
D3_GOMP_SPINCOUNT=10000 bash run.sh env_probe.py --tag spin10000 > logs/environment-spin10000.log 2>&1
D3_CPU_TUNE=a72 bash run.sh env_probe.py --tag a72 > logs/environment-a72.log 2>&1
bash run.sh env_probe.py --tag default-after > logs/environment-default-after.log 2>&1
