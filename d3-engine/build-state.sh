#!/usr/bin/env bash
set -euo pipefail
sources=${1:?directory containing state_scan.c, scan_prepare.c and conv_btc.c}
out=${2:?output library path}
gcc -O3 -std=c11 -march=armv8-a+simd -ffp-contract=off -fopenmp -Wall -Wextra -Werror -fPIC -shared \
    "$sources/state_scan.c" "$sources/scan_prepare.c" "$sources/conv_btc.c" -lm -o "$out"
