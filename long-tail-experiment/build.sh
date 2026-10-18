#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd tiled_attention.c -lm -o libtiled_attention.so
for threads in 2 3; do
 gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd -DD3_THREADS=$threads tuned_scan.c -lm -o "libtuned_scan$threads.so"
done
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd recurrent_scan.c -lm -o librecurrent_scan.so
gcc -Wall -Wextra -O3 -fPIC -shared -fopenmp -march=armv8-a+simd validity.c -lm -o libvalidity.so
