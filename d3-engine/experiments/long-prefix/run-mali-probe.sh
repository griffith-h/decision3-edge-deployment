#!/usr/bin/env bash
# Run manually on the board. Arguments are local paths; the image and runtime must already exist.
set -euo pipefail

patched=$(cd "${1:?patched d3-engine directory}" && pwd)
captures=$(cd "${2:?directory containing questions.pt}" && pwd)
runtime=$(cd "${3:?extracted GPU runtime root}" && pwd)
dest=$(cd "${4:?existing results directory}" && pwd)
image=${5:?local image ID from the benchmark}
shift 5
if (( $# > 1 )) || { (( $# == 1 )) && [[ $1 != --tiled ]]; }; then
    echo "The optional probe argument is --tiled" >&2
    exit 2
fi
study=$(cd "$(dirname "$0")" && pwd)
edge=$(cd "$study/../../../cpu-tail-experiment" && pwd)
run=$(mktemp -d "$dest/mali-qk.XXXXXXXX")
name="d3-$(basename "$run")"
printf 'Writing probe files to %s\n' "$run"

mkdir -p "$run/vendors"
printf '%s\n' /runtime/usr/lib/aarch64-linux-gnu/libRusticlOpenCL.so.1 > "$run/vendors/rusticl.icd"
taskset -c 0 g++ -O3 -std=c++17 -I "$edge" -I "$runtime/usr/include" "$study/probe_mali.cpp" \
    -L "$runtime/usr/lib/aarch64-linux-gnu" -lclblast -l:libOpenCL.so.1 -o "$run/qk"

isolated=(--rm --pull=never --network none --cpuset-cpus 0-7 --read-only --tmpfs /tmp --ulimit core=0)
docker run "${isolated[@]}" --name "$name-input" --memory 2g --memory-swap 2g \
    -v "$captures:/captures:ro" -v "$patched:/bench:ro" -v "$run:/out" \
    --entrypoint python "$image" -B /bench/probe_prefix.py /captures /out/qk.bin --mali-input

docker run "${isolated[@]}" --name "$name-gpu" --memory 4g --memory-swap 4g --device /dev/dri \
    -e LIBGL_DRIVERS_PATH=/board/lib/dri -e OCL_ICD_VENDORS=/vendors -e RUSTICL_ENABLE=panfrost \
    -e MESA_SHADER_CACHE_DIR=/tmp/mesa -v "$run:/bench:ro" -v "$edge/gpu_kernels.cl:/opt/kernels.cl:ro" \
    -v "$run/vendors:/vendors:ro" -v "$runtime:/runtime:ro" -v "$runtime/usr/lib/clc:/usr/lib/clc:ro" \
    -v "$runtime/usr/lib/llvm-19/lib/clang/19/include:/usr/lib/llvm-19/lib/clang/19/include:ro" \
    -v /usr/lib/aarch64-linux-gnu:/board/lib:ro --entrypoint /board/lib/ld-linux-aarch64.so.1 \
    "$image" --library-path /runtime/usr/lib/aarch64-linux-gnu:/board/lib \
    /bench/qk /opt/kernels.cl /bench/qk.bin "$@" > "$run/mali.jsonl.partial"
mv "$run/mali.jsonl.partial" "$run/mali.jsonl"
