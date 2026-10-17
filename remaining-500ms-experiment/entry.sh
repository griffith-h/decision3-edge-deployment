#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == *.sh ]]; then exec bash "/work/$1" "${@:2}"; fi
exec python -u "/work/${1:-diagnose.py}" "${@:2}"
