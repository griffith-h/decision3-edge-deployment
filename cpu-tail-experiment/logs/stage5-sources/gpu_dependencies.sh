#!/usr/bin/env bash
# Extract only. Never apt install; never modify /usr or host ICD configuration.
set -euo pipefail
lab=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$lab/gpu-debs" "$lab/gpu-root/vendors"
cd "$lab/gpu-debs"
apt-get download mesa-opencl-icd=25.0.7-2+deb13u1 libclc-19=1:19.1.7-3 libclang-cpp19=1:19.1.7-3+b1 libllvmspirvlib19.1 opencl-c-headers=3.0~2024.10.24-2 libclang-common-19-dev=1:19.1.7-3+b1
for pkg in ./*.deb; do dpkg-deb -x "$pkg" "$lab/gpu-root"; done
printf '%s\n' "$lab/gpu-root/usr/lib/aarch64-linux-gnu/libRusticlOpenCL.so.1" > "$lab/gpu-root/vendors/rusticl.icd"
