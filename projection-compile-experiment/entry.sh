#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --compiler ]]; then
 shift
 # Disposable container writable layer only; no docker build, no production changes.
 apt-get update -qq
 # docker-clean's update hook clears /var/cache/apt/archives, so restore after update.
 if [[ -d /work/apt-cache ]]; then cp /work/apt-cache/*.deb /var/cache/apt/archives/; fi
 apt-get install -y -qq --no-install-recommends g++ >/work/logs/compiler-install.log 2>&1
fi
if [[ ${1:-} == *.sh ]]; then exec bash "/work/$1" "${@:2}"; fi
exec python -u "/work/${1:-bench.py}" "${@:2}"
