#!/usr/bin/env bash
set -euo pipefail
sources=${1:?directory containing state_scan.c, scan_prepare.c and conv_btc.c}
out=${2:?output library path}
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
for source in scan_prepare conv_btc; do
    sed 's/num_threads(3)//g; s/schedule(static)/schedule(dynamic, 64)/g' "$sources/$source.c" > "$scratch/$source.c"
done
gcc -O3 -std=c11 -march=armv8-a+simd -ffp-contract=off -fopenmp -Wall -Wextra -Werror -fPIC -shared \
    "$sources/state_scan.c" "$scratch/scan_prepare.c" "$scratch/conv_btc.c" -lm -o "$out"
