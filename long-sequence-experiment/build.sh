#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd long_attention.c -lm -o liblong_attention.so
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd long_scan.c -lm -o liblong_scan.so
