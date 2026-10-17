#!/usr/bin/env bash
set -euo pipefail
lab=$(cd "$(dirname "$0")" && pwd)
g++ -Wall -Wextra -O3 -std=c++17 -fPIC -shared -I "$lab/gpu-root/usr/include" "$lab/gpu_host.cpp" -l:libOpenCL.so.1 -o "$lab/libgpu_host.so"
