#!/usr/bin/env bash
set -euo pipefail
if [[ -n $(docker ps --filter name=decision3-fusion-pipeline-test --format '{{.Names}}') ]]; then
 echo 'Finish the isolated test before replacing its libraries.' >&2
 exit 1
fi
cd -- "$(dirname -- "$0")"
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd -ffp-contract=off fused_conv.c -o libfused_conv.so
