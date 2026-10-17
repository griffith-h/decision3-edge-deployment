#!/usr/bin/env bash
# Host only. Private mount namespace; no system package installation.
set -euo pipefail
lab=/opt/decision3-edge-cpu-tail-20261017
mkdir -p "$lab/gpu-root/usr-upper" "$lab/gpu-root/usr-work" "$lab/gpu-cache"
exec unshare --mount --propagation private bash -c '
 set -euo pipefail
 lab=/opt/decision3-edge-cpu-tail-20261017
 mount -t overlay overlay -o lowerdir=/usr,upperdir=$lab/gpu-root/usr-upper,workdir=$lab/gpu-root/usr-work /usr
 mkdir -p /usr/lib/clc /usr/lib/llvm-19/lib/clang/19/include
 mount --bind "$lab/gpu-root/usr/lib/clc" /usr/lib/clc
 mount --bind "$lab/gpu-root/usr/lib/llvm-19/lib/clang/19/include" /usr/lib/llvm-19/lib/clang/19/include
 export LD_LIBRARY_PATH=$lab/gpu-root/usr/lib/aarch64-linux-gnu OCL_ICD_VENDORS=$lab/gpu-root/vendors RUSTICL_ENABLE=panfrost MESA_SHADER_CACHE_DIR=$lab/gpu-cache
 exec "$@"
' bash "$@"
