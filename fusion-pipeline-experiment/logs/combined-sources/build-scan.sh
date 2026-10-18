#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd fused_scan.c -o libfused_scan.so
