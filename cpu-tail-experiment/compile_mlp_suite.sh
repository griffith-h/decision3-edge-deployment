#!/usr/bin/env bash
set -euo pipefail
for layer in $(seq 0 11); do
 /work/compiler-env/bin/python /work/export_mlp.py --layer "$layer" > "/work/logs/mlp-export-$layer.log" 2>&1
 /work/compiler-env/bin/python /work/compile_mlp.py "mlp-$layer-fp16graph" > "/work/logs/mlp-compile-$layer.log" 2>&1
 python3 - "$layer" <<'PY'
import json,sys
r=json.load(open('/work/logs/compile-mlp-'+sys.argv[1]+'-fp16graph.json'))
assert r.get('export_return')==0,r
PY
done
